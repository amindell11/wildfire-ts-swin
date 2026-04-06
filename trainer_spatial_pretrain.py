"""
Training loop for spatial inpainting pre-training of SwinUnet.

Masks a large spatial block across all channels and reconstructs it.
Saves encoder+decoder weights for downstream fire prediction fine-tuning.
"""
import glob as _glob
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.wildfire.pretrain_dataset import PretrainDataset


# Channels to visualize
_VIS_CHANNELS = {
    'NDVI': 3,
    'Temp_Max': 9,
    'Elevation': 14,
    'PDSI': 15,
    'Fcst_Temp': 36,
}


def _log_reconstruction_vis(writer, x_orig, x_masked, pred, mask, epoch):
    """Log original / masked input / reconstruction for one sample."""
    for name, ch in _VIS_CHANNELS.items():
        def _norm(t):
            lo, hi = t.min(), t.max()
            if hi - lo < 1e-8:
                return torch.zeros_like(t)
            return (t - lo) / (hi - lo)

        grid = torch.cat([
            _norm(x_orig[ch].unsqueeze(0)),
            _norm(x_masked[ch].unsqueeze(0)),
            _norm(pred[ch].unsqueeze(0)),
        ], dim=1)  # (1, 3*H, W)
        writer.add_image(f'pretrain_vis/{name}', grid, epoch)


def _make_loader(dataset, batch_size, num_workers, shuffle=True, pin_memory=True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=shuffle,
        persistent_workers=(num_workers > 0),
    )


def trainer_spatial_pretrain(args, model, snapshot_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pin_memory = str(device) != 'cpu'

    # Logging
    log = logging.getLogger(f"pretrain_{id(snapshot_path)}")
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        fh = logging.FileHandler(os.path.join(snapshot_path, "pretrain_log.txt"))
        fh.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s',
                                          datefmt='%H:%M:%S'))
        log.addHandler(fh)

    def _log(msg):
        log.info(msg)
        tqdm.write(msg)

    _log(str(args))

    # Dataset with train/val split
    crops_per_tif = getattr(args, 'crops_per_tif', 4)
    base_dataset = PretrainDataset(
        tif_dir=args.tif_dir,
        crop_side_length=getattr(args, 'crop_side_length', 128),
        stats_years=getattr(args, 'stats_years', (2020, 2021)),
        crops_per_tif=1,
    )
    n_tifs = len(base_dataset)
    val_frac = getattr(args, 'val_frac', 0.1)
    n_val_tifs = max(1, int(n_tifs * val_frac))
    n_train_tifs = n_tifs - n_val_tifs

    gen = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(n_tifs, generator=gen).tolist()
    train_tif_paths = [base_dataset.tif_paths[i] for i in sorted(all_indices[:n_train_tifs])]
    val_tif_paths = [base_dataset.tif_paths[i] for i in sorted(all_indices[n_train_tifs:])]

    train_dataset = PretrainDataset(
        tif_dir=args.tif_dir,
        crop_side_length=getattr(args, 'crop_side_length', 128),
        stats_years=getattr(args, 'stats_years', (2020, 2021)),
        crops_per_tif=crops_per_tif,
    )
    train_dataset.tif_paths = train_tif_paths

    val_dataset = PretrainDataset(
        tif_dir=args.tif_dir,
        crop_side_length=getattr(args, 'crop_side_length', 128),
        stats_years=getattr(args, 'stats_years', (2020, 2021)),
        crops_per_tif=1,
    )
    val_dataset.tif_paths = val_tif_paths

    _log(f"Pre-training: {n_train_tifs} train TIFs x {crops_per_tif} crops = {len(train_dataset)} samples, "
         f"{n_val_tifs} val TIFs = {len(val_dataset)} samples")

    train_loader = _make_loader(train_dataset, args.batch_size, args.num_workers,
                                pin_memory=pin_memory)
    val_loader = _make_loader(val_dataset, args.batch_size, args.num_workers,
                              shuffle=False, pin_memory=pin_memory)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=getattr(args, 'weight_decay', 0.05),
        betas=getattr(args, 'betas', (0.9, 0.95)),
    )

    # LR schedule: linear warmup + cosine decay
    warmup_epochs = getattr(args, 'warmup_epochs', 10)
    max_epochs = args.max_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoint_interval = getattr(args, 'checkpoint_interval', 20)
    best_val_loss = float('inf')
    start_epoch = 0
    writer = SummaryWriter(os.path.join(snapshot_path, 'pretrain_log'))

    # Auto-resume from latest checkpoint
    ckpts = sorted(_glob.glob(os.path.join(snapshot_path, 'pretrain_ckpt_epoch*.pth')))
    if ckpts and not getattr(args, 'no_resume', False):
        latest_ckpt = ckpts[-1]
        _log(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        _log(f"  -> Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    epoch_bar = tqdm(range(start_epoch, max_epochs), desc="Pre-train", unit="ep", ncols=90)
    for epoch in epoch_bar:
        # ---- Train ----
        model.train()
        train_loss = 0.0

        batch_bar = tqdm(train_loader, desc="  Train", unit="batch", leave=False, ncols=90)
        for x_batch in batch_bar:
            x_batch = x_batch.to(device)

            loss, pred, mask = model(x_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch in val_loader:
                x_batch = x_batch.to(device)
                loss, _, _ = model(x_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        writer.add_scalar('pretrain/train_loss', train_loss, epoch)
        writer.add_scalar('pretrain/val_loss', val_loss, epoch)
        writer.add_scalar('pretrain/lr', optimizer.param_groups[0]['lr'], epoch)

        star = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.get_encoder_decoder_state_dict(),
                       os.path.join(snapshot_path, 'pretrain_best.pth'))
            star = " <-- best"

        _log(f"Epoch {epoch:03d}  train={train_loss:.6f}  val={val_loss:.6f}  "
             f"lr={optimizer.param_groups[0]['lr']:.2e}{star}")
        epoch_bar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}",
                              best=f"{best_val_loss:.6f}")

        # Periodic checkpoint + visualization
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(snapshot_path, f'pretrain_ckpt_epoch{epoch:03d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, ckpt_path)
            _log(f"  -> Checkpoint saved: {ckpt_path}")

            # Visualize reconstruction on one val sample
            with torch.no_grad():
                vis_x = next(iter(val_loader))[:1].to(device)
                vis_loss, vis_pred, vis_mask = model(vis_x)
                x_masked_vis = vis_x * (1.0 - vis_mask)
                _log_reconstruction_vis(
                    writer,
                    vis_x[0].cpu(),
                    x_masked_vis[0].cpu(),
                    vis_pred[0].cpu(),
                    vis_mask[0].cpu(),
                    epoch,
                )

    # Save final transferable weights
    final_path = os.path.join(snapshot_path, 'pretrain_encoder_decoder.pth')
    torch.save(model.get_encoder_decoder_state_dict(), final_path)
    _log(f"Pre-training done. Best val loss: {best_val_loss:.6f}")
    _log(f"Transferable weights saved to: {final_path}")

    writer.close()
    return best_val_loss
