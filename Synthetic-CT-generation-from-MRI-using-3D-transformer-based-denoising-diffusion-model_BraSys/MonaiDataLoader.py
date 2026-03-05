import os
import json
import random
import numpy as np
import nibabel as nib
import torch

from os import listdir
from os.path import join
from typing import Hashable, Mapping, Sequence, Tuple
from collections.abc import Hashable as ABC_Hashable, Mapping as ABC_Mapping, Sequence as ABC_Sequence

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import ITKReader, CacheDataset, Dataset
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import pad_list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged,
    RandAffined, RandGaussianNoised, RandAdjustContrastd, RandBiasFieldd,
    RandShiftIntensityd, RandScaleIntensityd, RandGridDistortiond,
    RandGaussianSmoothd, RandSpatialCropSamplesd, CropForegroundd,
    DeleteItemsd, NormalizeIntensityd, ToTensord, EnsureType,
    CopyItemsd, ResizeWithPadOrCropd, SpatialCropd,
    LoadImage, EnsureChannelFirst, Orientation, CropForeground,
    NormalizeIntensity, ClipIntensityPercentiles, ClipIntensityPercentilesd, ScaleIntensityd
)
from monai.transforms import MapTransform, Randomizable, Transform
from monai.transforms.transform import MapTransform as CoreMapTransform, RandomizableTransform
from monai.utils import set_determinism
from monai.utils.enums import TransformBackends

def worker_init_fn(worker_id):
    # Grab the PyTorch seed (already different per worker under the hood)
    # and fold it into a 32‑bit NumPy/Python seed.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

class CreateForegroundMask(Transform):
    """
    Creates a binary mask that defines the foreground based on thresholds in RGB or HSV color space.
    This transform receives an RGB (or grayscale) image where by default it is assumed that the foreground has
    low values (dark) while the background has high values (white). Otherwise, set `invert` argument to `True`.

    Args:
        threshold: an int or a float number that defines the threshold that values less than that are foreground.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        threshold: float  = 0.0
    ) -> None:
        self.threshold = threshold

    def __call__(self, t1,t1c,t2,flair):     
        # Create a boolean mask that's True wherever any modality > 0
        foreground_mask_bool = (t1 > 0) | (t1c > 0) | (t2 > 0) | (flair > 0)

        # Convert to uint8 (0/1) if you need an integer mask
        final_foreground_mask = foreground_mask_bool.astype(np.uint8)

        return final_foreground_mask

class CreateForegroundMaskd(MapTransform):
    """
    Creates a binary mask that defines the foreground based on threshold and unhealthy mask.

    Args:
        keys: keys of the corresponding items to be transformed. Must be the mri followed by the unhealthy mask

    """

    def __init__(
        self,
        keys: KeysCollection,
        threshold: float = 0.0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = CreateForegroundMask(threshold=threshold)
        self.keys = keys

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        d["foreground_mask"] = self.transform(
            t1=d[self.keys[0]],
            t1c=d[self.keys[1]],
            t2=d[self.keys[2]],
            flair=d[self.keys[3]]
            )
        return d



class MonaiDataLoader():
    def __init__(self,
        spatial_size,
        number_of_patches,
        cache_rate,
        num_workers,
        prob,
        use_global_stats,
        batch_size=2,
        json_file='/homes/andre.ferreira/BraTS2025/Dataset/Global-Synthesis/data_split.json',
    ):
        self.spatial_size = spatial_size
        self.number_of_patches = number_of_patches
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.prob = prob
        self.use_global_stats = use_global_stats
        with open(json_file, 'r') as f:
            self.data_split = json.load(f)
        if use_global_stats:
            with open(json_file.replace('data_split.json', 'stats.json'), 'r') as file:
                self.stats_data = json.load(file)
    
    def get_aug_transforms(self):
        keys_list = ['t1', 't1c', 't2', 'flair']  
        mode_list = ["trilinear", "trilinear", "trilinear", "trilinear"]
        aug_transforms = [
            ###############################
            ###### DATA AUGMENTATION ######
            # Based on https://arxiv.org/pdf/2006.06676.pdf
            # rotate 3 degrees
            # translation (16,32,8)
            # scale_range (-0.1, 0.1) -> zoom!
            # shear_range (-0.1, 0.1)
            RandAffined(
                keys=keys_list,
                prob=self.prob,
                rotate_range=((-np.pi/60,np.pi/60),(-np.pi/60,np.pi/60),(-np.pi/60,np.pi/60)), # 3 degrees
                #translate_range=(16,16,4), 
                scale_range=((-0.03,0.03),(-0.03,0.03),(-0.03,0.03)),
                shear_range=((-0.05,0.05),(-0.05,0.05),(-0.05,0.05)),
                padding_mode="border",
                mode=mode_list,
                ),
            # RandGridDistortiond (Elastic Deformation)
            RandGridDistortiond(keys=keys_list, prob=self.prob, num_cells=(5, 5, 5), distort_limit=(0.01, 0.01, 0.01), padding_mode="border", mode=mode_list),
            
            # Bias field
            RandBiasFieldd(keys=[keys_list[0]], degree=3, coeff_range=(0.0, 0.05), prob=self.prob),

            #### Intensity #### 
            # Simulate Rand Gamma Image
            RandShiftIntensityd(keys=[keys_list[0]], prob=self.prob, offsets=(-0.1, 0.1)),
            RandScaleIntensityd(keys=[keys_list[0]], prob=self.prob, factors=(-0.1, 0.1)),
            RandAdjustContrastd(keys=[keys_list[0]], prob=self.prob, gamma=(0.9, 1.1)),
            # Blur
            RandGaussianSmoothd(keys=[keys_list[0]], sigma_x=(0.1, 0.4), sigma_y=(0.1, 0.4), sigma_z=(0.1, 0.4), approx='erf', prob=self.prob, allow_missing_keys=False),
            #Noise gaussian
            RandGaussianNoised(keys=[keys_list[0]], prob=self.prob/2, mean=0, std=0.01),
            ### FINISH DATA AUGMENTATION ##
            ###############################
        ]
        return aug_transforms

    def get_io_transforms(self, dtype: torch.dtype = torch.float32):
        keys_list = ['t1', 't1c', 't2', 'flair']  

        # Load data to memory transforms
        io_transforms = [
            LoadImaged(keys=keys_list, image_only=True, reader=ITKReader()),
            EnsureChannelFirstd(keys=keys_list),
            Orientationd(keys=keys_list, axcodes="RAS"),
            CreateForegroundMaskd(keys=keys_list),   
            CropForegroundd(keys=keys_list+['foreground_mask'], source_key='foreground_mask'),
            ToTensord(keys=keys_list+['foreground_mask'])
        ]
        return io_transforms
    
    def get_out_transforms(self, dtype: torch.dtype = torch.float32):
        keys_list = ['t1', 't1c', 't2', 'flair']  

        # Load data to memory transforms
        out_transforms = [
                ClipIntensityPercentilesd(
                                keys=keys_list,
                                lower=0.1,
                                upper=99.9
                                )
            ]

        if self.use_global_stats:
            print(f"Using global stats")
            for modality in keys_list:
                mean = self.stats_data[modality]['mean']
                std = self.stats_data[modality]['std']
                out_transforms.append(
                    NormalizeIntensityd(
                                keys=[modality],
                                subtrahend=mean,
                                divisor=std,
                                nonzero=False,
                                channel_wise=False
                                )
                    )

        else:
            out_transforms.append(
                NormalizeIntensityd(
                            keys=keys_list,
                            subtrahend=None,
                            divisor=None,
                            nonzero=False,
                            channel_wise=True
                            )
                )
        out_transforms.append(
            ScaleIntensityd(
                            keys=keys_list,
                            minv=-1.0,
                            maxv=1.0,
                            channel_wise=True,
                            )
            )
        out_transforms.append(
                RandSpatialCropSamplesd(
                            keys=keys_list+['foreground_mask'],
                            roi_size=self.spatial_size,
                            num_samples=self.number_of_patches,
                            random_size=False
                            )
            )
        out_transforms.append(
                ResizeWithPadOrCropd(
                            keys=keys_list+['foreground_mask'],
                            spatial_size=self.spatial_size,
                            mode="minimum")
            )
        out_transforms.append(DeleteItemsd(keys=['foreground_start_coord', 'foreground_end_coord']))
        out_transforms.append(ToTensord(keys=keys_list+['foreground_mask']))
                                        
        return out_transforms

    def get_loaders_train(self):
        io_transforms = self.get_io_transforms()
        aug_transforms = self.get_aug_transforms()
        out_transforms = self.get_out_transforms()

        cached_ds_mri = CacheDataset(
            data=self.data_split['train_cases_mri'],
            transform=Compose(io_transforms),
            cache_rate=self.cache_rate,   # only deterministic steps cached
            num_workers=self.num_workers,
        )

        aug_dataset = Dataset(
            data=cached_ds_mri,
            transform=Compose(aug_transforms),
        )
        out_dataset = Dataset(
            data=aug_dataset,
            transform=Compose(out_transforms),
        )
        
        train_loader_mri = DataLoader(
                out_dataset,
                batch_size=self.batch_size,
                shuffle=True,                 # let DataLoader shuffle
                generator=torch.Generator(),  # we’ll reseed it each epoch
                num_workers=self.num_workers,
                persistent_workers=True,      # keep workers alive across epochs
                pin_memory=True,
                prefetch_factor=2,
                collate_fn=pad_list_data_collate,
                worker_init_fn=worker_init_fn
        )
        return train_loader_mri
    
    def get_out_transforms_val(self, dtype: torch.dtype = torch.float32):
        keys_list = ['t1', 't1c', 't2', 'flair']  

        # Load data to memory transforms
        # Load data to memory transforms
        out_transforms = [
                ClipIntensityPercentilesd(
                                keys=keys_list,
                                lower=0.1,
                                upper=99.9
                                )
            ]

        if self.use_global_stats:
            print(f"Using global stats")
            for modality in keys_list:
                mean = self.stats_data[modality]['mean']
                std = self.stats_data[modality]['std']
                out_transforms.append(
                    NormalizeIntensityd(
                                keys=[modality],
                                subtrahend=mean,
                                divisor=std,
                                nonzero=False,
                                channel_wise=False
                                )
                    )

        else:
            out_transforms.append(
                NormalizeIntensityd(
                            keys=keys_list,
                            subtrahend=None,
                            divisor=None,
                            nonzero=False,
                            channel_wise=True
                            )
                )

        out_transforms.append(
            ScaleIntensityd(
                        keys=keys_list,
                        minv=-1.0,
                        maxv=1.0,
                        channel_wise=True,
                        )
            )
        out_transforms.append(
            CopyItemsd(
                    keys=keys_list+['foreground_mask'],
                    times=1,
                    names=['t1_fullres', 't1c_fullres', 't2_fullres', 'flair_fullres', 'foreground_mask_fullres']
                    )
            )
        out_transforms.append(
            RandSpatialCropSamplesd(
                            keys=keys_list+['foreground_mask'],
                            roi_size=self.spatial_size,
                            num_samples=self.number_of_patches,
                            random_size=False
                            )
            )
        out_transforms.append(
            ResizeWithPadOrCropd(
                            keys=keys_list+['foreground_mask'],
                            spatial_size=self.spatial_size,
                            mode="minimum")
            )
        out_transforms.append(
            DeleteItemsd(keys=['foreground_start_coord', 'foreground_end_coord'])
            )
        out_transforms.append(
            ToTensord(keys=keys_list+['foreground_mask']+['t1_fullres', 't1c_fullres', 't2_fullres', 'flair_fullres', 'foreground_mask_fullres'])
            )
                                    
        return out_transforms

    def get_loaders_val(self):
        io_transforms = self.get_io_transforms()
        out_transforms = self.get_out_transforms_val()


        cached_ds_mri = CacheDataset(
            data=self.data_split['val_cases_mri'],
            transform=Compose(io_transforms),
            cache_rate=self.cache_rate,   # only deterministic steps cached
            num_workers=self.num_workers,
        )

        final_ds_mri = Dataset(
            data=cached_ds_mri,
            transform=Compose(out_transforms),
        )
         
        val_loader_mri = DataLoader(
                final_ds_mri,
                batch_size=1,
                shuffle=True,                 # let DataLoader shuffle
                generator=torch.Generator(),  # we’ll reseed it each epoch
                num_workers=self.num_workers,
                persistent_workers=True,      # keep workers alive across epochs
                pin_memory=True,
                prefetch_factor=2,
                collate_fn=pad_list_data_collate,
                worker_init_fn=worker_init_fn
        )
        return val_loader_mri