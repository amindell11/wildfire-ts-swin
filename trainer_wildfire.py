"""
Training loop for SwinUnet on the WildfireSpreadTS dataset.

Loss: Focal Loss (alpha = inverse fire-pixel frequency, gamma = 2.0).
  - Fire pixels are very rare (~1% of all pixels); focal loss down-weights
    easy examples and focuses on hard positives.

Best model is selected by validation AP (Average Precision) on the fire class.
"""
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.wildfire import WildfireDataset, get_year_split
from utils import FocalLoss, compute_binary_metrics, compute_ap


def _make_loader(dataset, batch_size, shuffle, num_workers, pin_memory=True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=shuffle,  # drop last incomplete batch only during training
    )


def trainer_wildfire(args, model, snapshot_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # XLA (TPU) support: pin_memory must be False and mark_step() is needed
    _xm = None
    try:
        import torch_xla.core.xla_model as xm
        if 'xla' in str(device):
            _xm = xm
    except ImportError:
        pass
    pin_memory = _xm is None and str(device) != 'cpu'

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    # Use tqdm.write for console so progress bars aren't garbled
    class _TqdmHandler(logging.StreamHandler):
        def emit(self, record):
            tqdm.write(self.format(record))
    logging.getLogger().addHandler(_TqdmHandler())
    logging.info(str(args))

    train_years, val_years, test_years = get_year_split(args.data_fold_id)
    logging.info(f"Train years: {train_years}  Val years: {val_years}  Test years: {test_years}")

    common_kwargs = dict(
        data_dir=args.data_dir,
        n_leading_observations=args.n_leading_observations,
        crop_side_length=args.crop_side_length,
        load_from_hdf5=args.load_from_hdf5,
    )

    db_train = WildfireDataset(
        **common_kwargs,
        included_fire_years=train_years,
        is_train=True,
        stats_years=train_years,
    )
    db_val = WildfireDataset(
        **common_kwargs,
        included_fire_years=val_years,
        is_train=False,  # center-crop only, no augmentation
        stats_years=train_years,  # normalise with train stats
        n_leading_observations_test_adjustment=args.n_leading_observations_test_adjustment,
    )
    logging.info(f"Train samples: {len(db_train)}  Val samples: {len(db_val)}")

    train_loader = _make_loader(db_train, args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader   = _make_loader(db_val,   args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin_memory)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # --- Loss ---
    # Focal loss: alpha = inverse fire-pixel frequency (paper convention).
    # Fire pixels ~1% → alpha_fire ~100, clipped to 50 for stability.
    fire_alpha = min(50.0, 1.0 / 0.01)
    alpha = torch.tensor([1.0, fire_alpha], dtype=torch.float32).to(device)
    focal_loss_fn = FocalLoss(alpha=alpha, gamma=args.focal_gamma)

    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.base_lr * 1e-2)

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    iter_num = 0
    best_val_ap = -1.0
    max_iterations = args.max_epochs * len(train_loader)

    epoch_bar = tqdm(range(args.max_epochs), desc="Epochs", unit="ep", ncols=90)
    for epoch in epoch_bar:
        # ---- Train ----
        model.train()
        train_loss = 0.0

        batch_bar = tqdm(train_loader, desc=f"  Train", unit="batch",
                         leave=False, ncols=90)
        for x_batch, y_batch in batch_bar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)                          # (B, 2, H, W)
            loss = focal_loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if _xm is not None:
                _xm.mark_step()

            # Poly-style LR within epoch (in addition to cosine between epochs)
            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = lr_

            train_loss += loss.item()
            writer.add_scalar('train/loss_focal', loss.item(), iter_num)
            iter_num += 1
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_:.2e}")

        train_loss /= len(train_loader)
        logging.info(f"Epoch {epoch:03d}  train_loss={train_loss:.4f}")

        scheduler.step()

        # ---- Validation ----
        if (epoch + 1) % args.eval_interval != 0:
            epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}")
            continue

        model.eval()
        val_focal = 0.0
        all_preds, all_probs, all_gts = [], [], []

        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"  Val  ",
                                         unit="batch", leave=False, ncols=90):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                val_focal += focal_loss_fn(logits, y_batch).item()

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(np.int64)
                gts   = y_batch.cpu().numpy()
                all_probs.append(probs.flatten())
                all_preds.append(preds.flatten())
                all_gts.append(gts.flatten())

        val_focal /= len(val_loader)
        val_loss = val_focal

        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_gts   = np.concatenate(all_gts)
        metrics = compute_binary_metrics(all_preds, all_gts)
        ap = compute_ap(all_probs, all_gts)

        star = " *" if ap > best_val_ap else ""
        logging.info(
            f"Epoch {epoch:03d}  val_loss={val_loss:.4f}  "
            f"AP={ap:.4f}  F1={metrics['f1']:.4f}  "
            f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}{star}"
        )
        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            AP=f"{ap:.4f}",
            F1=f"{metrics['f1']:.4f}",
            best=f"{max(ap, best_val_ap):.4f}",
        )

        writer.add_scalar('val/loss',      val_loss,            epoch)
        writer.add_scalar('val/ap',        ap,                  epoch)
        writer.add_scalar('val/f1',        metrics['f1'],       epoch)
        writer.add_scalar('val/iou',       metrics['iou'],      epoch)
        writer.add_scalar('val/precision', metrics['precision'], epoch)
        writer.add_scalar('val/recall',    metrics['recall'],   epoch)

        if ap > best_val_ap:
            best_val_ap = ap
            ckpt_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"  -> New best AP={best_val_ap:.4f}, saved to {ckpt_path}")
        else:
            ckpt_path = os.path.join(snapshot_path, 'last_model.pth')
            torch.save(model.state_dict(), ckpt_path)

    writer.close()
    logging.info(f"Training finished. Best val AP: {best_val_ap:.4f}")
    return "Training Finished!"
