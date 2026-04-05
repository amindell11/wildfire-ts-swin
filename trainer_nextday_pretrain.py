"""
Training loop for next-day prediction pre-training of SwinUnet.

Input: today's satellite/weather features. Target: tomorrow's features.
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


from networks.nextday_swin_unet import DYNAMIC_CHANNELS

# Dynamic channels to visualize: (name, index in DYNAMIC_CHANNELS output)
_VIS_DYNAMIC = {
    'NDVI': (3, DYNAMIC_CHANNELS.index(3)),
    'Temp_Max': (9, DYNAMIC_CHANNELS.index(9)),
    'Wind_Spd': (6, DYNAMIC_CHANNELS.index(6)),
    'Precip': (5, DYNAMIC_CHANNELS.index(5)),
    'Fcst_Temp': (36, DYNAMIC_CHANNELS.index(36)),
}


def _log_reconstruction_vis(writer, x_today, y_tomorrow, pred, epoch):
    """Log today / actual tomorrow / predicted tomorrow for dynamic channels."""
    for name, (input_ch, pred_ch) in _VIS_DYNAMIC.items():
        def _norm(t):
            lo, hi = t.min(), t.max()
            if hi - lo < 1e-8:
                return torch.zeros_like(t)
            return (t - lo) / (hi - lo)

        grid = torch.cat([
            _norm(x_today[input_ch].unsqueeze(0)),      # today's value
            _norm(y_tomorrow[input_ch].unsqueeze(0)),    # actual tomorrow
            _norm(pred[pred_ch].unsqueeze(0)),           # predicted tomorrow
        ], dim=1)  # (1, 3*H, W)
        writer.add_image(f'pretrain_vis/{name}', grid, epoch)


def _make_loader(dataset, batch_size, num_workers, shuffle=True, pin_memory=True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=shuffle,
        persistent_workers=(num_workers > 0),
    )


def trainer_nextday_pretrain(args, model, snapshot_path, device=None):
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

    # Dataset with train/val split by location
    crops_per_location = getattr(args, 'crops_per_location', 1)
    base_dataset = PretrainDataset(
        tif_dir=args.tif_dir,
        crop_side_length=getattr(args, 'crop_side_length', 128),
        stats_years=getattr(args, 'stats_years', (2020, 2021)),
        crops_per_location=1,
    )
    n_locs = len(base_dataset)
    val_frac = getattr(args, 'val_frac', 0.1)
    n_val_locs = max(1, int(n_locs * val_frac))
    n_train_locs = n_locs - n_val_locs

    gen = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(n_locs, generator=gen).tolist()
    train_loc_indices = sorted(all_indices[:n_train_locs])
    val_loc_indices = sorted(all_indices[n_train_locs:])

    train_pairs = [base_dataset.location_pairs[i] for i in train_loc_indices]
    val_pairs = [base_dataset.location_pairs[i] for i in val_loc_indices]

    train_dataset = PretrainDataset(
        tif_dir=args.tif_dir,
        crop_side_length=getattr(args, 'crop_side_length', 128),
        stats_years=getattr(args, 'stats_years', (2020, 2021)),
        crops_per_location=crops_per_location,
    )
    train_dataset.location_pairs = train_pairs

    val_dataset = PretrainDataset(
        tif_dir=args.tif_dir,
        crop_side_length=getattr(args, 'crop_side_length', 128),
        stats_years=getattr(args, 'stats_years', (2020, 2021)),
        crops_per_location=1,
    )
    val_dataset.location_pairs = val_pairs

    _log(f"Pre-training: {n_train_locs} train locations x {crops_per_location} crops = {len(train_dataset)} samples, "
         f"{n_val_locs} val locations = {len(val_dataset)} samples")

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

    # Auto-resume from latest checkpoint (disable with args.no_resume=True)
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
        for x_batch, y_batch in batch_bar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss, pred = model(x_batch, y_batch)

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
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss, _ = model(x_batch, y_batch)
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

            # Visualize: today / actual tomorrow / predicted tomorrow
            with torch.no_grad():
                vis_x, vis_y = next(iter(val_loader))
                vis_x, vis_y = vis_x[:1].to(device), vis_y[:1].to(device)
                _, vis_pred = model(vis_x, vis_y)
                _log_reconstruction_vis(
                    writer,
                    vis_x[0].cpu(),
                    vis_y[0].cpu(),
                    vis_pred[0].cpu(),
                    epoch,
                )

    # Save final transferable weights
    final_path = os.path.join(snapshot_path, 'pretrain_encoder_decoder.pth')
    torch.save(model.get_encoder_decoder_state_dict(), final_path)
    _log(f"Pre-training done. Best val loss: {best_val_loss:.6f}")
    _log(f"Transferable weights saved to: {final_path}")

    writer.close()
    return best_val_loss
