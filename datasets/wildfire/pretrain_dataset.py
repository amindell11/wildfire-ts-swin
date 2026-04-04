"""
PretrainDataset: reads single-day GeoTIFF files for self-supervised pre-training.

Each TIF has 19 raw bands (same as WildfireSpreadTS but with active fire = 0).
Preprocessing mirrors FireSpreadDataset.preprocess_and_augment to produce 40
normalized channels, but without fire-aware crop weighting or label extraction.

Output per sample: (40, crop_side_length, crop_side_length) — no label.
"""
import glob
import warnings
from typing import List, Tuple

import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .dataloader_utils import get_means_stds_missing_values, get_indices_of_degree_features


class PretrainDataset(Dataset):
    def __init__(
        self,
        tif_dir: str,
        crop_side_length: int = 128,
        stats_years: Tuple[int, int] = (2020, 2021),
        crops_per_tif: int = 1,
    ):
        self.crop_side_length = crop_side_length
        self.crops_per_tif = crops_per_tif

        self.tif_paths = sorted(glob.glob(f"{tif_dir}/**/*.tif", recursive=True))
        if not self.tif_paths:
            raise FileNotFoundError(f"No .tif files found under {tif_dir}")

        # Filter out TIFs that are too small to crop
        valid_paths = []
        for p in self.tif_paths:
            try:
                with rasterio.open(p, 'r') as ds:
                    if ds.height >= crop_side_length and ds.width >= crop_side_length:
                        valid_paths.append(p)
            except Exception:
                pass
        skipped = len(self.tif_paths) - len(valid_paths)
        if skipped > 0:
            warnings.warn(f"Skipped {skipped} TIFs smaller than {crop_side_length}x{crop_side_length}")
        self.tif_paths = valid_paths

        means, stds, _ = get_means_stds_missing_values(list(stats_years))
        self.means = torch.from_numpy(means).float()[None, :, None, None]   # (1, 23, 1, 1)
        self.stds = torch.from_numpy(stds).float()[None, :, None, None]
        self.indices_of_degree_features = get_indices_of_degree_features()   # [7, 13, 19]
        self.one_hot_matrix = torch.eye(17)

    def __len__(self):
        return len(self.tif_paths) * self.crops_per_tif

    def __getitem__(self, index):
        tif_index = index // self.crops_per_tif
        with rasterio.open(self.tif_paths[tif_index], 'r') as ds:
            img = ds.read().astype(np.float32)  # (19, H, W)

        # Add temporal dimension: (1, 19, H, W)
        x = torch.from_numpy(img).unsqueeze(0)

        # --- Preprocessing (mirrors FireSpreadDataset.preprocess_and_augment) ---

        # NaN handling for active fire band (last band)
        x[:, -1, ...] = torch.nan_to_num(x[:, -1, ...], nan=0)
        x[:, -1, ...] = torch.floor_divide(x[:, -1, ...], 100)

        # Random crop
        _, _, H, W = x.shape
        top = np.random.randint(0, H - self.crop_side_length + 1)
        left = np.random.randint(0, W - self.crop_side_length + 1)
        x = TF.crop(x, top, left, self.crop_side_length, self.crop_side_length)

        # Random augmentation (flip / rotate) with degree feature adjustment
        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))

        if hflip:
            x = TF.hflip(x)
            x[:, self.indices_of_degree_features, ...] = (
                360 - x[:, self.indices_of_degree_features, ...])

        if vflip:
            x = TF.vflip(x)
            x[:, self.indices_of_degree_features, ...] = (
                180 - x[:, self.indices_of_degree_features, ...]) % 360

        if rotate != 0:
            angle = rotate * 90
            x = TF.rotate(x, angle)
            x[:, self.indices_of_degree_features, ...] = (
                x[:, self.indices_of_degree_features, ...] - 90 * rotate) % 360

        # Sin-transform degree features
        x[:, self.indices_of_degree_features, ...] = torch.sin(
            torch.deg2rad(x[:, self.indices_of_degree_features, ...]))

        # Binary active fire mask (before standardization)
        binary_af_mask = (x[:, -1:, ...] > 0).float()

        # Z-score standardization
        x = (x - self.means) / self.stds

        # Append binary AF mask → 24 channels
        x = torch.cat([x, binary_af_mask], dim=1)
        x = torch.nan_to_num(x, nan=0.0)

        # One-hot encode landcover (channel 16) → 17 channels
        # x shape: (1, 24, H, W) → (1, 40, H, W)
        new_shape = (x.shape[0], x.shape[2], x.shape[3], self.one_hot_matrix.shape[0])
        landcover_classes = x[:, 16, ...].long().flatten() - 1
        landcover_classes = landcover_classes.clamp(0, 16)
        landcover_encoding = self.one_hot_matrix[landcover_classes].reshape(
            new_shape).permute(0, 3, 1, 2)
        x = torch.cat([x[:, :16, ...], landcover_encoding, x[:, 17:, ...]], dim=1)

        # Squeeze temporal dimension → (40, H, W)
        return x.squeeze(0)
