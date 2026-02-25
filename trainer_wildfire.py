"""
Training loop for SwinUnet on the WildfireSpreadTS dataset.

Loss: weighted CrossEntropy + Dice on the fire class.
  - Fire pixels are very rare (~1% of all pixels), so CE is weighted inversely
    by class frequency to avoid the model predicting all-no-fire.
  - Dice is computed only on the fire class (class 1).

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
from tqdm import tqdm

from datasets.wildfire import WildfireDataset, get_year_split
from utils import DiceLoss, compute_binary_metrics, compute_ap


def _make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        drop_last=shuffle,  # drop last incomplete batch only during training
    )


def trainer_wildfire(args, model, snapshot_path):
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
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
                                num_workers=args.num_workers)
    val_loader   = _make_loader(db_val,   args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # --- Loss ---
    # CE weight: upweight the fire class relative to no-fire.
    # A rough heuristic: fire pixels are ~1% of total → weight ~[1, 99].
    # Clipped to [1, 50] to avoid instability early in training.
    fire_weight = min(50.0, 1.0 / 0.01)
    ce_weights = torch.tensor([1.0, fire_weight], dtype=torch.float32).cuda()
    ce_loss_fn   = nn.CrossEntropyLoss(weight=ce_weights)
    dice_loss_fn = DiceLoss(n_classes=2)

    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.base_lr * 1e-2)

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    iter_num = 0
    best_val_ap = -1.0
    max_iterations = args.max_epochs * len(train_loader)

    for epoch in tqdm(range(args.max_epochs), ncols=70):
        # ---- Train ----
        model.train()
        train_ce = train_dice = 0.0

        for x_batch, y_batch in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            logits = model(x_batch)                          # (B, 2, H, W)
            loss_ce   = ce_loss_fn(logits, y_batch)
            loss_dice = dice_loss_fn(logits, y_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Poly-style LR within epoch (in addition to cosine between epochs)
            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = lr_

            train_ce   += loss_ce.item()
            train_dice += loss_dice.item()
            writer.add_scalar('train/loss_ce',   loss_ce.item(),   iter_num)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iter_num)
            iter_num += 1

        train_ce   /= len(train_loader)
        train_dice /= len(train_loader)
        train_loss  = 0.4 * train_ce + 0.6 * train_dice
        logging.info(
            f"Train epoch {epoch}: loss={train_loss:.4f}  CE={train_ce:.4f}  Dice={train_dice:.4f}")

        scheduler.step()

        # ---- Validation ----
        if (epoch + 1) % args.eval_interval != 0:
            continue

        model.eval()
        val_ce = val_dice = 0.0
        all_preds, all_probs, all_gts = [], [], []

        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

                logits = model(x_batch)
                val_ce   += ce_loss_fn(logits, y_batch).item()
                val_dice += dice_loss_fn(logits, y_batch, softmax=True).item()

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(np.int64)
                gts   = y_batch.cpu().numpy()
                all_probs.append(probs.flatten())
                all_preds.append(preds.flatten())
                all_gts.append(gts.flatten())

        val_ce   /= len(val_loader)
        val_dice /= len(val_loader)
        val_loss  = 0.4 * val_ce + 0.6 * val_dice

        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_gts   = np.concatenate(all_gts)
        metrics = compute_binary_metrics(all_preds, all_gts)
        ap = compute_ap(all_probs, all_gts)

        logging.info(
            f"Val   epoch {epoch}: loss={val_loss:.4f}  CE={val_ce:.4f}  Dice={val_dice:.4f}  "
            f"AP={ap:.4f}  F1={metrics['f1']:.4f}  IoU={metrics['iou']:.4f}  "
            f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}"
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
