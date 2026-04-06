"""
PretrainDataset: reads consecutive-day GeoTIFF pairs for next-day prediction pre-training.

Each location folder contains 2 TIFs (day1, day2). The dataset loads both,
preprocesses them identically to FireSpreadDataset, and returns (day1, day2)
as input/target pairs. The model learns to predict tomorrow's features from today's.

Output per sample: (x, y) where x = (40, H, W) day1, y = (40, H, W) day2.
"""
import glob
import os
import warnings
from typing import Tuple

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
        crops_per_location: int = 1,
    ):
        self.crop_side_length = crop_side_length
        self.crops_per_location = crops_per_location

        # Find location folders that contain exactly 2 TIF files
        self.location_pairs = []
        for root, dirs, files in os.walk(tif_dir):
            tifs = sorted([f for f in files if f.endswith('.tif')])
            if len(tifs) >= 2:
                # Take the first two chronologically (sorted by date filename)
                self.location_pairs.append((
                    os.path.join(root, tifs[0]),
                    os.path.join(root, tifs[1]),
                ))

        if not self.location_pairs:
            raise FileNotFoundError(
                f"No location folders with 2+ TIF files found under {tif_dir}")

        # Filter out pairs where either TIF is too small or has wrong band count
        valid_pairs = []
        for p1, p2 in self.location_pairs:
            try:
                with rasterio.open(p1, 'r') as ds1, rasterio.open(p2, 'r') as ds2:
                    if (ds1.count == 23 and ds2.count == 23
                            and ds1.height >= crop_side_length and ds1.width >= crop_side_length
                            and ds2.height >= crop_side_length and ds2.width >= crop_side_length):
                        valid_pairs.append((p1, p2))
            except Exception:
                pass
        skipped = len(self.location_pairs) - len(valid_pairs)
        if skipped > 0:
            warnings.warn(f"Skipped {skipped} locations (TIFs too small or unreadable)")
        self.location_pairs = valid_pairs

        means, stds, _ = get_means_stds_missing_values(list(stats_years))
        self.means = torch.from_numpy(means).float()[None, :, None, None]
        self.stds = torch.from_numpy(stds).float()[None, :, None, None]
        self.indices_of_degree_features = get_indices_of_degree_features()
        self.one_hot_matrix = torch.eye(17)

    def __len__(self):
        return len(self.location_pairs) * self.crops_per_location

    def _read_tif(self, path):
        with rasterio.open(path, 'r') as ds:
            return ds.read().astype(np.float32)  # (19, H, W)

    def _preprocess(self, x, top, left, hflip, vflip, rotate):
        """Preprocess a single day: 19 raw bands -> 40 normalized channels.
        Uses shared crop/augmentation params so both days get identical spatial transforms.
        """
        # NaN handling for active fire band (last band)
        x[:, -1, ...] = torch.nan_to_num(x[:, -1, ...], nan=0)
        x[:, -1, ...] = torch.floor_divide(x[:, -1, ...], 100)

        # Crop (same location for both days)
        x = TF.crop(x, top, left, self.crop_side_length, self.crop_side_length)

        # Augmentation (same transforms for both days)
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

        # Binary active fire mask
        binary_af_mask = (x[:, -1:, ...] > 0).float()

        # Z-score standardization
        x = (x - self.means) / self.stds

        # Append binary AF mask -> 24 channels
        x = torch.cat([x, binary_af_mask], dim=1)
        x = torch.nan_to_num(x, nan=0.0)

        # One-hot encode landcover (channel 16) -> 17 channels -> 40 total
        new_shape = (x.shape[0], x.shape[2], x.shape[3], self.one_hot_matrix.shape[0])
        landcover_classes = x[:, 16, ...].long().flatten() - 1
        landcover_classes = landcover_classes.clamp(0, 16)
        landcover_encoding = self.one_hot_matrix[landcover_classes].reshape(
            new_shape).permute(0, 3, 1, 2)
        x = torch.cat([x[:, :16, ...], landcover_encoding, x[:, 17:, ...]], dim=1)

        return x.squeeze(0)  # (40, H, W)

    def __getitem__(self, index):
        loc_index = index // self.crops_per_location
        path1, path2 = self.location_pairs[loc_index]

        # Read both days
        img1 = torch.from_numpy(self._read_tif(path1)).unsqueeze(0)  # (1, 19, H, W)
        img2 = torch.from_numpy(self._read_tif(path2)).unsqueeze(0)

        # Generate shared spatial augmentation params
        _, _, H, W = img1.shape
        top = np.random.randint(0, H - self.crop_side_length + 1)
        left = np.random.randint(0, W - self.crop_side_length + 1)
        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))

        # Preprocess both with same spatial transforms
        x = self._preprocess(img1, top, left, hflip, vflip, rotate)  # (40, H, W)
        y = self._preprocess(img2, top, left, hflip, vflip, rotate)  # (40, H, W)

        return x, y
