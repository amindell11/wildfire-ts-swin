"""
FireSpreadDataset from WildfireSpreadTS (https://github.com/SebastianGer/WildfireSpreadTS).
Original MIT License. Adapted for use in this repo (relative import changed to absolute).
"""
from pathlib import Path
from typing import List, Optional

import rasterio
from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import warnings
from .dataloader_utils import get_means_stds_missing_values, get_indices_of_degree_features
import torchvision.transforms.functional as TF
import h5py
from datetime import datetime


class FireSpreadDataset(Dataset):
    def __init__(self, data_dir: str, included_fire_years: List[int], n_leading_observations: int,
                 crop_side_length: int, load_from_hdf5: bool, is_train: bool, remove_duplicate_features: bool,
                 stats_years: List[int], n_leading_observations_test_adjustment: Optional[int] = None,
                 features_to_keep: Optional[List[int]] = None, return_doy: bool = False):
        """
        Args:
            data_dir: Root directory of the dataset, containing folders per fire year.
            included_fire_years: Years to include in this dataset instance.
            n_leading_observations: Number of days to use as input.
            crop_side_length: Side length of random square crops during training.
            load_from_hdf5: If True, load from HDF5 files instead of TIF.
            is_train: If True, apply geometric augmentations; otherwise center-crop only.
            remove_duplicate_features: Remove static features from all timesteps but the last,
                then flatten temporal dimension.
            stats_years: Years used to compute normalization statistics.
            n_leading_observations_test_adjustment: Skip initial samples to align test set
                across different n_leading_observations values.
            features_to_keep: Indices (0–39) of features to keep; None = keep all.
            return_doy: If True, also return day-of-year per timestep.
        """
        super().__init__()

        self.stats_years = stats_years
        self.return_doy = return_doy
        self.features_to_keep = features_to_keep
        self.remove_duplicate_features = remove_duplicate_features
        self.is_train = is_train
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.included_fire_years = included_fire_years
        self.data_dir = data_dir

        self.validate_inputs()

        if self.n_leading_observations_test_adjustment is None:
            self.skip_initial_samples = 0
        else:
            self.skip_initial_samples = self.n_leading_observations_test_adjustment - self.n_leading_observations
            if self.skip_initial_samples < 0:
                raise ValueError(
                    f"n_leading_observations_test_adjustment must be >= n_leading_observations, "
                    f"got {self.n_leading_observations_test_adjustment=} and {self.n_leading_observations=}"
                )

        self.imgs_per_fire = self.read_list_of_images()
        self.datapoints_per_fire = self.compute_datapoints_per_fire()
        self.length = sum(
            sum(self.datapoints_per_fire[fy].values())
            for fy in self.datapoints_per_fire
        )

        self.one_hot_matrix = torch.eye(17)
        self.means, self.stds, _ = get_means_stds_missing_values(self.stats_years)
        self.means = self.means[None, :, None, None]
        self.stds = self.stds[None, :, None, None]
        self.indices_of_degree_features = get_indices_of_degree_features()

    def find_image_index_from_dataset_index(self, target_id):
        if target_id < 0:
            target_id = self.length + target_id
        if target_id >= self.length:
            raise RuntimeError(f"Tried to access item {target_id}, max index is {self.length - 1}.")

        first_id_in_current_fire = 0
        found_fire_year = None
        found_fire_name = None
        for fire_year in self.datapoints_per_fire:
            if found_fire_year is None:
                for fire_name, datapoints_in_fire in self.datapoints_per_fire[fire_year].items():
                    if target_id - first_id_in_current_fire < datapoints_in_fire:
                        found_fire_year = fire_year
                        found_fire_name = fire_name
                        break
                    else:
                        first_id_in_current_fire += datapoints_in_fire

        in_fire_index = target_id - first_id_in_current_fire
        return found_fire_year, found_fire_name, in_fire_index

    def load_imgs(self, found_fire_year, found_fire_name, in_fire_index):
        """Returns (x, y) or (x, y, doys).
        x: (n_leading_observations, n_features, H, W)
        y: (H, W) binary next-day fire mask
        """
        in_fire_index += self.skip_initial_samples
        end_index = in_fire_index + self.n_leading_observations + 1

        if self.load_from_hdf5:
            hdf5_path = self.imgs_per_fire[found_fire_year][found_fire_name][0]
            with h5py.File(hdf5_path, 'r') as f:
                imgs = f["data"][in_fire_index:end_index]
                if self.return_doy:
                    doys = f["data"].attrs["img_dates"][in_fire_index:(end_index - 1)]
                    doys = self.img_dates_to_doys(doys)
                    doys = torch.Tensor(doys)
            x, y = np.split(imgs, [-1], axis=0)
            y = y[0, -1, ...]
        else:
            imgs_to_load = self.imgs_per_fire[found_fire_year][found_fire_name][in_fire_index:end_index]
            imgs = []
            for img_path in imgs_to_load:
                with rasterio.open(img_path, 'r') as ds:
                    imgs.append(ds.read())
            x = np.stack(imgs[:-1], axis=0)
            y = imgs[-1][-1, ...]

        if self.return_doy:
            return x, y, doys
        return x, y

    def __getitem__(self, index):
        found_fire_year, found_fire_name, in_fire_index = self.find_image_index_from_dataset_index(index)
        loaded_imgs = self.load_imgs(found_fire_year, found_fire_name, in_fire_index)

        if self.return_doy:
            x, y, doys = loaded_imgs
        else:
            x, y = loaded_imgs

        x, y = self.preprocess_and_augment(x, y)

        if self.remove_duplicate_features and self.n_leading_observations > 1:
            x = self.flatten_and_remove_duplicate_features_(x)
        elif self.features_to_keep is not None:
            if len(x.shape) != 4:
                raise NotImplementedError(
                    f"Removing features is only implemented for 4D tensors, got {x.shape=}.")
            x = x[:, self.features_to_keep, ...]

        if self.return_doy:
            return x, y, doys
        return x, y

    def __len__(self):
        return self.length

    def validate_inputs(self):
        if self.n_leading_observations < 1:
            raise ValueError("Need at least one day of observations.")
        if self.return_doy and not self.load_from_hdf5:
            raise NotImplementedError("Returning day of year is only implemented for HDF5 files.")
        if self.n_leading_observations_test_adjustment is not None:
            if self.n_leading_observations_test_adjustment < self.n_leading_observations:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be >= n_leading_observations.")
            if self.n_leading_observations_test_adjustment < 1:
                raise ValueError("n_leading_observations_test_adjustment must be >= 1.")

    def read_list_of_images(self):
        imgs_per_fire = {}
        for fire_year in self.included_fire_years:
            imgs_per_fire[fire_year] = {}
            if not self.load_from_hdf5:
                fires_in_year = glob.glob(f"{self.data_dir}/{fire_year}/*/")
                fires_in_year.sort()
                for fire_dir_path in fires_in_year:
                    fire_name = fire_dir_path.split("/")[-2]
                    fire_img_paths = glob.glob(f"{fire_dir_path}/*.tif")
                    fire_img_paths.sort()
                    imgs_per_fire[fire_year][fire_name] = fire_img_paths
                    if len(fire_img_paths) == 0:
                        warnings.warn(
                            f"Fire {fire_year}: {fire_name} contains no images.", RuntimeWarning)
            else:
                fires_in_year = glob.glob(f"{self.data_dir}/{fire_year}/*.hdf5")
                fires_in_year.sort()
                for fire_hdf5 in fires_in_year:
                    fire_name = Path(fire_hdf5).stem
                    imgs_per_fire[fire_year][fire_name] = [fire_hdf5]
        return imgs_per_fire

    def compute_datapoints_per_fire(self):
        datapoints_per_fire = {}
        for fire_year in self.imgs_per_fire:
            datapoints_per_fire[fire_year] = {}
            for fire_name, fire_imgs in self.imgs_per_fire[fire_year].items():
                if not self.load_from_hdf5:
                    n_fire_imgs = len(fire_imgs) - self.skip_initial_samples
                else:
                    if not fire_imgs:
                        n_fire_imgs = 0
                    else:
                        with h5py.File(fire_imgs[0], 'r') as f:
                            n_fire_imgs = len(f["data"]) - self.skip_initial_samples
                datapoints_in_fire = n_fire_imgs - self.n_leading_observations
                if datapoints_in_fire <= 0:
                    warnings.warn(
                        f"Fire {fire_year}: {fire_name} contributes no data points "
                        f"({len(fire_imgs)} images, lead={self.n_leading_observations}).",
                        RuntimeWarning)
                    datapoints_per_fire[fire_year][fire_name] = 0
                else:
                    datapoints_per_fire[fire_year][fire_name] = datapoints_in_fire
        return datapoints_per_fire

    def standardize_features(self, x):
        return (x - self.means) / self.stds

    def preprocess_and_augment(self, x, y):
        x, y = torch.Tensor(x), torch.Tensor(y)

        if not self.load_from_hdf5:
            x[:, -1, ...] = torch.nan_to_num(x[:, -1, ...], nan=0)
            y = torch.nan_to_num(y, nan=0.0)
            x[:, -1, ...] = torch.floor_divide(x[:, -1, ...], 100)

        y = (y > 0).long()

        if self.is_train:
            x, y = self.augment(x, y)
        else:
            x, y = self.center_crop_x32(x, y)

        x[:, self.indices_of_degree_features, ...] = torch.sin(
            torch.deg2rad(x[:, self.indices_of_degree_features, ...]))

        binary_af_mask = (x[:, -1:, ...] > 0).float()
        x = self.standardize_features(x)
        x = torch.cat([x, binary_af_mask], axis=1)
        x = torch.nan_to_num(x, nan=0.0)

        new_shape = (x.shape[0], x.shape[2], x.shape[3], self.one_hot_matrix.shape[0])
        landcover_classes_flattened = x[:, 16, ...].long().flatten() - 1
        landcover_encoding = self.one_hot_matrix[landcover_classes_flattened].reshape(
            new_shape).permute(0, 3, 1, 2)
        x = torch.concatenate([x[:, :16, ...], landcover_encoding, x[:, 17:, ...]], dim=1)

        return x, y

    def augment(self, x, y):
        best_n_fire_pixels = -1
        best_crop = (None, None)
        for _ in range(10):
            top = np.random.randint(0, x.shape[-2] - self.crop_side_length)
            left = np.random.randint(0, x.shape[-1] - self.crop_side_length)
            x_crop = TF.crop(x, top, left, self.crop_side_length, self.crop_side_length)
            y_crop = TF.crop(y, top, left, self.crop_side_length, self.crop_side_length)
            n_fire_pixels = x_crop[:, -1, ...].mean() + 1000 * y_crop.float().mean()
            if n_fire_pixels > best_n_fire_pixels:
                best_n_fire_pixels = n_fire_pixels
                best_crop = (x_crop, y_crop)
        x, y = best_crop

        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))

        if hflip:
            x = TF.hflip(x)
            y = TF.hflip(y)
            x[:, self.indices_of_degree_features, ...] = 360 - x[:, self.indices_of_degree_features, ...]

        if vflip:
            x = TF.vflip(x)
            y = TF.vflip(y)
            x[:, self.indices_of_degree_features, ...] = (
                180 - x[:, self.indices_of_degree_features, ...]) % 360

        if rotate != 0:
            angle = rotate * 90
            x = TF.rotate(x, angle)
            y = TF.rotate(y.unsqueeze(0), angle).squeeze(0)
            x[:, self.indices_of_degree_features, ...] = (
                x[:, self.indices_of_degree_features, ...] - 90 * rotate) % 360

        return x, y

    def center_crop_x32(self, x, y):
        crop = (self.crop_side_length // 32) * 32
        x = TF.center_crop(x, (crop, crop))
        y = TF.center_crop(y, (crop, crop))
        return x, y

    def flatten_and_remove_duplicate_features_(self, x):
        static_feature_ids, dynamic_feature_ids = self.get_static_and_dynamic_features_to_keep(
            self.features_to_keep)
        dynamic_feature_ids = torch.tensor(dynamic_feature_ids).int()
        x_dynamic_only = x[:-1, dynamic_feature_ids, :, :].flatten(start_dim=0, end_dim=1)
        x_last_day = x[-1, self.features_to_keep, ...].squeeze(0)
        return torch.cat([x_dynamic_only, x_last_day], axis=0)

    @staticmethod
    def get_static_and_dynamic_feature_ids():
        static_feature_ids = [12, 13, 14] + list(range(16, 33))
        dynamic_feature_ids = list(range(12)) + [15] + list(range(33, 40))
        return static_feature_ids, dynamic_feature_ids

    @staticmethod
    def get_static_and_dynamic_features_to_keep(features_to_keep: Optional[List[int]]):
        static_features, dynamic_features = FireSpreadDataset.get_static_and_dynamic_feature_ids()
        if isinstance(features_to_keep, list):
            dynamic_features = sorted(set(dynamic_features) & set(features_to_keep))
            static_features = sorted(set(static_features) & set(features_to_keep))
        return static_features, dynamic_features

    @staticmethod
    def get_n_features(n_observations: int, features_to_keep: Optional[List[int]],
                       deduplicate_static_features: bool):
        static_features, dynamic_features = FireSpreadDataset.get_static_and_dynamic_features_to_keep(
            features_to_keep)
        n_static = len(static_features)
        n_dynamic = len(dynamic_features)
        n_all = n_static + n_dynamic
        n_features = (int(deduplicate_static_features) * n_dynamic) * (n_observations - 1) + n_all
        return n_features

    @staticmethod
    def img_dates_to_doys(img_dates):
        date_format = "%Y-%m-%d"
        return [datetime.strptime(d.replace(".tif", ""), date_format).timetuple().tm_yday
                for d in img_dates]

    @staticmethod
    def map_channel_index_to_features(only_base: bool = False):
        base_feature_names = [
            'VIIRS band M11', 'VIIRS band I2', 'VIIRS band I1', 'NDVI', 'EVI2',
            'Total precipitation', 'Wind speed', 'Wind direction',
            'Minimum temperature', 'Maximum temperature', 'Energy release component',
            'Specific humidity', 'Slope', 'Aspect', 'Elevation',
            'Palmer drought severity index (PDSI)', 'Landcover class',
            'Forecast: Total precipitation', 'Forecast: Wind speed', 'Forecast: Wind direction',
            'Forecast: Temperature', 'Forecast: Specific humidity', 'Active fire',
        ]
        land_cover_classes = [
            'Land cover: Evergreen Needleleaf Forests', 'Land cover: Evergreen Broadleaf Forests',
            'Land cover: Deciduous Needleleaf Forests', 'Land cover: Deciduous Broadleaf Forests',
            'Land cover: Mixed Forests', 'Land cover: Closed Shrublands',
            'Land cover: Open Shrublands', 'Land cover: Woody Savannas',
            'Land cover: Savannas', 'Land cover: Grasslands', 'Land cover: Permanent Wetlands',
            'Land cover: Croplands', 'Land cover: Urban and Built-up Lands',
            'Land cover: Cropland/Natural Vegetation Mosaics',
            'Land cover: Permanent Snow and Ice', 'Land cover: Barren', 'Land cover: Water Bodies',
        ]
        if only_base:
            return dict(enumerate(base_feature_names))
        return_features = (base_feature_names[:16] + land_cover_classes
                           + base_feature_names[17:] + ["Active fire (binary)"])
        return dict(enumerate(return_features))
