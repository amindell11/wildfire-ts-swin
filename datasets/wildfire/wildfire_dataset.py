"""
WildfireDataset: wraps FireSpreadDataset for SwinUnet training.

FireSpreadDataset returns x of shape (T, C, H, W). This wrapper either:
  - Passes it through intact for FactoredPatchEmbed (use_factored_embed=True)
  - Flattens to (T*C, H, W) for the standard PatchEmbed (use_factored_embed=False)

Example with n_leading_observations=3:
  raw x shape : (3, 40, H, W)
  factored    : (3, 40, H, W) — fed to SwinUnet with in_chans=40, n_timesteps=3
  flattened   : (120, H, W)   — fed to SwinUnet with in_chans=120
  output y    : (H, W)  long  — binary fire mask (0=no fire, 1=fire)
"""
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from .FireSpreadDataset import FireSpreadDataset

# Number of feature channels per timestep produced by FireSpreadDataset after preprocessing:
#   16 numeric features + 17 one-hot land cover + 5 forecast + 1 AF time + 1 binary AF mask = 40
N_FEATURES_PER_TIMESTEP = 40

# Train / val / test year splits (12 folds rotating over 2018-2021).
# Each entry is (train_year_1, train_year_2, val_year, test_year).
DATA_FOLDS = [
    (2018, 2019, 2020, 2021), (2018, 2019, 2021, 2020),
    (2018, 2020, 2019, 2021), (2018, 2020, 2021, 2019),
    (2018, 2021, 2019, 2020), (2018, 2021, 2020, 2019),
    (2019, 2020, 2018, 2021), (2019, 2020, 2021, 2018),
    (2019, 2021, 2018, 2020), (2019, 2021, 2020, 2018),
    (2020, 2021, 2018, 2019), (2020, 2021, 2019, 2018),
]


def get_year_split(data_fold_id: int):
    fold = DATA_FOLDS[data_fold_id]
    train_years = list(fold[:2])
    val_years   = list(fold[2:3])
    test_years  = list(fold[3:4])
    return train_years, val_years, test_years


class WildfireDataset(Dataset):
    """Thin wrapper around FireSpreadDataset that handles temporal dimension formatting."""

    def __init__(
        self,
        data_dir: str,
        included_fire_years: List[int],
        n_leading_observations: int,
        crop_side_length: int,
        load_from_hdf5: bool,
        is_train: bool,
        stats_years: List[int],
        n_leading_observations_test_adjustment: Optional[int] = None,
        features_to_keep: Optional[List[int]] = None,
        use_factored_embed: bool = True,
    ):
        self.inner = FireSpreadDataset(
            data_dir=data_dir,
            included_fire_years=included_fire_years,
            n_leading_observations=n_leading_observations,
            crop_side_length=crop_side_length,
            load_from_hdf5=load_from_hdf5,
            is_train=is_train,
            remove_duplicate_features=False,  # keep full (T, C, H, W) for channel concatenation
            stats_years=stats_years,
            n_leading_observations_test_adjustment=n_leading_observations_test_adjustment,
            features_to_keep=features_to_keep,
            return_doy=False,
        )
        self.n_leading_observations = n_leading_observations
        self.n_channels = N_FEATURES_PER_TIMESTEP
        self.use_factored_embed = use_factored_embed

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        x, y = self.inner[index]
        # x: (T, C, H, W) from FireSpreadDataset
        if self.use_factored_embed:
            # keep temporal dimension intact for FactoredPatchEmbed
            return x, y.long()
        else:
            # flatten to (T*C, H, W) for standard PatchEmbed
            T, C, H, W = x.shape
            return x.reshape(T * C, H, W), y.long()
