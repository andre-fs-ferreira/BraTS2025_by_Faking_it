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
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged, ScaleIntensity,
    RandAffined, RandGaussianNoised, RandAdjustContrastd, RandBiasFieldd,
    RandShiftIntensityd, RandScaleIntensityd, RandGridDistortiond,
    RandGaussianSmoothd, RandSpatialCropSamplesd, CropForegroundd,
    DeleteItemsd, NormalizeIntensityd, ToTensord, EnsureType,
    CopyItemsd, ResizeWithPadOrCropd, SpatialCropd,
    LoadImage, EnsureChannelFirst, Orientation, CropForeground,
    NormalizeIntensity, ClipIntensityPercentiles
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

    def __call__(self, mri, unhealthy_mask):     
        foreground_mask_bool = mri > 0
        foreground_mask = foreground_mask_bool.astype(np.uint8)

        healthy_regions_mask = (unhealthy_mask == 0).astype(np.uint8)
        final_foreground_mask = foreground_mask * healthy_regions_mask

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
        d["healthy_region_mask"] = self.transform(
            mri=d[self.keys[0]],
            unhealthy_mask=d[self.keys[1]]
            )
        return d

class RandSelectRandomPointd(MapTransform, Randomizable):
    """
    Randomly select a point inside a binary mask and store it under a new key.

    Args:
        keys: keys of the corresponding mask items to be transformed (e.g. ['healthy_region_mask']).
        allow_missing_keys: don't raise error if key is missing.
        seed: random seed for reproducibility (optional).
    """
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        seed: int = None
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if seed is not None:
            self.set_random_state(seed)

    def randomize(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: Tensor shape [C, X, Y, Z] or [1, X, Y, Z]
        m = mask
        indices = np.argwhere(m == 1)
        if indices.shape[0] == 0:
            raise ValueError("No foreground voxel found to sample.")
        idx = self.R.randint(indices.shape[0])
        point = indices[idx].tolist()
        return tuple(point)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            mask = d[key]
            # generate random point
            point = self.randomize(mask)
            # store as a tensor under key + '_point'
            d[f"{key}_point"] = torch.tensor(point, dtype=torch.long)
        return d

class RandCorrectRandomMaskd(MapTransform, Randomizable):
    """
    Inserts a smaller healthy mask patch into a larger unhealthy volume at a random (or provided) center,
    avoiding overlap with existing unhealthy tissue.

    Args:
        keys: sequence of three keys in the input mapping:
            [mask_unhealthy_key, mask_healthy_key, center_key]
        allow_missing_keys: if True, skip missing keys without error.
        seed: optional seed for reproducibility.
    """
    def __init__(
        self,
        keys: KeysCollection,
        healthMasks,
        healthMasksLoader,
        allow_missing_keys: bool = False,
        seed: int = None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        # Expect keys = ["mask_unhealthy", "mask_healthy", "mask_point"]
        if seed is not None:
            self.set_random_state(seed)
        self.healthMasks = healthMasks
        self.healthMasksLoader=healthMasksLoader
    def __call__(
        self,
        data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        for i in range(100):
            d = dict(data)
            um_key, pt_key = self.keys # unhealthy mask and random center
            mask_unhealthy = d[um_key]        
            center = d[pt_key]

            # Random Healthy mask
            healthMasks_path = random.choice(self.healthMasks)["healthy_mask"]
            mask_healthy = self.healthMasksLoader(healthMasks_path)

            _, W, H, D     = mask_unhealthy.shape
            _, hX, hY, hZ = mask_healthy.shape
            _, x0, y0, z0       = center

            # start with an empty “healthy” volume
            corrected = torch.zeros_like(mask_unhealthy)

            # compute the target slice in the full volume
            x_start = x0 - hX // 2;  x_end = x_start + hX
            y_start = y0 - hY // 2;  y_end = y_start + hY
            z_start = z0 - hZ // 2;  z_end = z_start + hZ

            # clamp to volume bounds
            x1, x2 = max(x_start, 0), min(x_end, W)
            y1, y2 = max(y_start, 0), min(y_end, H)
            z1, z2 = max(z_start, 0), min(z_end, D)

            # compute how much of the patch we actually use
            cx1 = x1 - x_start;  cx2 = cx1 + (x2 - x1)
            cy1 = y1 - y_start;  cy2 = cy1 + (y2 - y1)
            cz1 = z1 - z_start;  cz2 = cz1 + (z2 - z1)

            # pull out sub‐volumes
            vol_sub   = corrected[0, x1:x2, y1:y2, z1:z2]
            patch_sub = mask_healthy[0, cx1:cx2, cy1:cy2, cz1:cz2]

            # only place patch where there’s no “unhealthy” tissue already
            allowed = (patch_sub == 1) & (mask_unhealthy[0,x1:x2,y1:y2,z1:z2] == 0)
            vol_sub[allowed] = 1

            # write back
            corrected[0, x1:x2, y1:y2, z1:z2] = vol_sub
            d[f"healthy_region_mask_corrected"] = corrected
            if torch.sum(corrected)>4:
                continue_flag = True
                break
        if continue_flag==True:
            return d
        else:
            raise RuntimeError(f"Could not find a valid crop after 100 tries (needed {self.min_mask_voxels} voxels)")

class ClipAndZNormalizationd(MapTransform):
    """
    Compute voided and target volumes, then clip + z‑normalize both.

    voided = mri * (1 - corrected_mask) * (1 - unhealthy_mask)
    target = mri * corrected_mask

    Args:
        keys: sequence of three keys:
            [mri_key, corrected_mask_key, unhealthy_mask_key]
        voided_key: name under which to store the voided output
        target_key: name under which to store the target output
        lower: lower percentile for clipping (default 0.1)
        upper: upper percentile for clipping (default 99.9)
        allow_missing_keys: if True, skip missing keys without error
    """
    def __init__(
        self,
        keys: KeysCollection,
        voided_key: str = "voided_mri",
        target_key: str = "target",
        lower: float = 0.1,
        upper: float = 99.9,
        intensity_scale_range: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.voided_key = voided_key
        self.target_key = target_key
        self.clipper = ClipIntensityPercentiles(
            lower=lower, upper=upper,
            sharpness_factor=None,
            channel_wise=False,
            return_clipping_values=False,
        )
        self.intensity_scale_range = intensity_scale_range

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        mri_key, corr_key, un_key = self.keys

        mri = d[mri_key]
        corr = d[corr_key]
        un = d[un_key]
        
        # compute voided and target
        voided = mri * (1 - corr) * (1 - un)
        target = mri * (1 - un) # Everyting exept the unhealthy tissue

        # clip intensities
        voided = self.clipper(voided)
        target = self.clipper(target)

        # compute stats on nonzero of voided
        mask = voided != 0
        vals = voided[mask]
        mean, std = float(vals.mean()), float(vals.std())

        # normalize both
        normalizer = NormalizeIntensity(
            subtrahend=mean,
            divisor=std,
            nonzero=False,
            channel_wise=False,
        )
        voided = normalizer(voided)
        target = normalizer(target)

        if self.intensity_scale_range:
            intensity_scaler = ScaleIntensity(
                minv=-1.0,
                maxv=1.0,
                )
            voided = intensity_scaler(voided)
            target = intensity_scaler(target)
                                   
        # write back
        d[self.voided_key] = voided
        d[self.target_key] = target
        d['mean'] = mean
        d['std'] = std
        return d

class RandPatchByMaskd(MapTransform):
    """
    Sample N patches of size `spatial_size` from (voided_mri, corrected_mask_healthy, target),
    each centered on a random point *within* `corrected_mask_healthy` (value==1), with an extra
    random shift of up to half a patch size, clamped to volume bounds, and rejecting any crop
    that has fewer than `min_mask_voxels` foreground voxels.

    Args:
        keys: list of three keys in the data mapping:
              [voided_mri_key, corrected_mask_key, target_key]
        spatial_size: tuple (px,py,pz) of the crop size
        num_patches: how many valid patches to sample per volume
        min_mask_voxels: minimum nonzero voxels in `corrected_mask_key` per crop
        allow_missing_keys: if True, skip missing keys without error
    """
    def __init__(
        self,
        keys: Sequence[str],
        spatial_size: Tuple[int,int,int],
        num_patches: int,
        min_mask_voxels: int = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.vox_key, self.mask_key, self.tgt_key = keys
        self.spatial_size = spatial_size
        self.num_patches = num_patches
        self.min_mask_voxels = min_mask_voxels

    def _select_random_center(self, mask: torch.Tensor) -> Tuple[int,int,int]:
        """Pick a random foreground voxel, then shift up to half patch in each dim."""
        # mask: [1,W,H,D] or [C,W,H,D]
        m = mask
        inds = torch.nonzero(m==1, as_tuple=False)
        if inds.shape[0] == 0:
            raise ValueError("No healthy voxel to sample center from")
        # pick one at random
        idx = torch.randint(0, inds.shape[0], (1,), device=inds.device).item()
        _, x0,y0,z0 = inds[idx].tolist()
        # random shift up to half patch
        hx, hy, hz = [s//2 for s in self.spatial_size]
        sx = np.random.randint(-hx, hx+1)
        sy = np.random.randint(-hy, hy+1)
        sz = np.random.randint(-hz, hz+1)
        x, y, z = x0+sx, y0+sy, z0+sz
        # clamp to bounds
        _, W,H,D = m.shape
        x = min(max(x, hx), W-hx)
        y = min(max(y, hy), H-hy)
        z = min(max(z, hz), D-hz)
        return x,y,z

    def _is_valid(self, mask_crop: torch.Tensor) -> bool:
        return int(torch.count_nonzero(mask_crop)) >= self.min_mask_voxels

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict:
        d = dict(data)
        vol = d[self.vox_key]
        mask = d[self.mask_key]
        tgt = d[self.tgt_key]

        patches = []
        for _ in range(self.num_patches):
            for _trial in range(100):
                x,y,z = self._select_random_center(mask)
                crop = SpatialCropd(
                    keys=[self.vox_key, self.mask_key, self.tgt_key],
                    roi_size=self.spatial_size,
                    roi_center=(x,y,z),
                )(d)
                if self._is_valid(crop[self.mask_key]):
                    patches.append(crop)
                    break
            else:
                raise RuntimeError(f"Could not find a valid crop after 100 tries (needed {self.min_mask_voxels} voxels)")

        # stack them along new batch dim
        for key in (self.vox_key, self.mask_key, self.tgt_key):
            stacked = torch.stack([p[key] for p in patches], dim=0)
            d[key] = stacked

        return d

class RandHealthyMaskLoader(MapTransform, Randomizable):
    def __init__(self,
        spatial_size,
        json_file='/homes/andre.ferreira/BraTS2025/Dataset/Local-Synthesis/data_split.json',
    ):
        self.spatial_size = spatial_size
        self.number_of_patches = number_of_patches
        self.min_mask_voxels = min_mask_voxels
        with open(json_file, 'r') as f:
            self.data_split = json.load(f)

    def get_healty_masks_transforms(self, dtype: torch.dtype = torch.float32):
        keys_list = ['healthy_mask'] 
        
        # Load data to memory transforms
        io_transforms = [
            LoadImaged(keys=keys_list, image_only=True, reader=ITKReader()),
            EnsureChannelFirstd(keys=keys_list),
            Orientationd(keys=keys_list, axcodes="RAS"),
            CropForegroundd(keys=keys_list, source_key='healthy_mask'),
            # DATA AUGMENTATION # 
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
            # Rotate
            RandFlipd(keys=keys_list, prob=self.prob),
            RandRotate90d(keys=keys_list, prob=self.prob),
            # FINISH DATA AUGMENTATION # 
            ToTensord(keys=keys_list, dtype=dtype, allow_missing_keys=False)
        ]
        return Compose(io_transforms)

    def get_mask_loader_train(self, cache_rate, num_workers):
        healthy_mask_io_transforms = self.get_healty_masks_transforms()
        cached_ds_mask = CacheDataset(
            data=self.data_split['train_masks'],
            transform=healthy_mask_io_transforms,
            cache_rate=cache_rate,   # only deterministic steps cached
            num_workers=num_workers,
        )
        train_loader_mask = DataLoader(
                cached_ds_mask,
                batch_size=1,
                shuffle=True,                 # let DataLoader shuffle
                generator=torch.Generator(),  # we’ll reseed it each epoch
                num_workers=num_workers,
                persistent_workers=True,      # keep workers alive across epochs
                pin_memory=True,
                prefetch_factor=2,
                collate_fn=pad_list_data_collate,
                worker_init_fn=worker_init_fn
        )
        return train_loader_mask

class MonaiDataLoader():
    def __init__(self,
        spatial_size,
        number_of_patches,
        cache_rate,
        num_workers,
        prob,
        batch_size=2,
        min_mask_voxels=4,
        intensity_scale_range=False,
        json_file='/homes/andre.ferreira/BraTS2025/Dataset/Local-Synthesis/data_split.json',
    ):
        self.spatial_size = spatial_size
        self.number_of_patches = number_of_patches
        self.min_mask_voxels = min_mask_voxels
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.prob = prob
        self.intensity_scale_range = intensity_scale_range

        with open(json_file, 'r') as f:
            self.data_split = json.load(f)

        self.clipping = ClipIntensityPercentiles(
            lower=0.1, 
            upper=99.9, 
            sharpness_factor=None, 
            channel_wise=False, 
            return_clipping_values=False
            )
                
        self.ensure_shape = ResizeWithPadOrCropd(
                            keys=['voided_mri', 'corrected_mask_healthy'],
                            spatial_size=spatial_size,
                            mode="minimum")
    def get_healty_masks_transforms(self):
        # Load data to memory transforms
        def threshold_at_one(x):
            # threshold at 1
            return x == 1
        io_transforms = [
            LoadImage(image_only=True, reader=ITKReader()),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            CropForeground(select_fn=threshold_at_one, margin=0),
        ]
        return Compose(io_transforms)
    
    def get_aug_transforms(self):
        keys_list = ['mri', 'healthy_region_mask']  
        mode_list = ["trilinear", "nearest"]
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
        keys_list = ['mri', 'unhealthy_mask'] 

        # Load data to memory transforms
        io_transforms = [
            LoadImaged(keys=keys_list, image_only=True, reader=ITKReader()),
            EnsureChannelFirstd(keys=keys_list),
            Orientationd(keys=keys_list, axcodes="RAS"),
            CreateForegroundMaskd(keys=keys_list),   
        ]
        random_transforms = [
                RandSelectRandomPointd(keys=['healthy_region_mask']),
                RandCorrectRandomMaskd(keys=['unhealthy_mask', 'healthy_region_mask_point'], healthMasks=self.data_split['train_masks'], healthMasksLoader=self.get_healty_masks_transforms()),
                ClipAndZNormalizationd(keys=['mri','healthy_region_mask_corrected', 'unhealthy_mask'], intensity_scale_range=self.intensity_scale_range),
                RandPatchByMaskd(
                    keys=["voided_mri","healthy_region_mask_corrected", "target"],
                    spatial_size=self.spatial_size,
                    num_patches=self.number_of_patches,
                    min_mask_voxels=4,
                ),
                ToTensord(keys=["voided_mri","healthy_region_mask_corrected","target",'mean','std'])
            ]
        return io_transforms, random_transforms

    def get_loaders_train(self):
        io_transforms, random_transforms = self.get_io_transforms()
        aug_transforms = self.get_aug_transforms()

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
        
        final_ds_mri = Dataset(
            data=aug_dataset,
            transform=Compose(random_transforms),
        )
         
        train_loader_mri = DataLoader(
                final_ds_mri,
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
    
    def get_io_transforms_val(self, dtype: torch.dtype = torch.float32):
        keys_list = ['mri', 'unhealthy_mask'] 

        # Load data to memory transforms
        io_transforms = [
            LoadImaged(keys=keys_list, image_only=True, reader=ITKReader()),
            EnsureChannelFirstd(keys=keys_list),
            Orientationd(keys=keys_list, axcodes="RAS"),
            CreateForegroundMaskd(keys=keys_list),   
        ]
        random_transforms = [
                RandSelectRandomPointd(keys=['healthy_region_mask']),
                RandCorrectRandomMaskd(keys=['unhealthy_mask', 'healthy_region_mask_point'], healthMasks=self.data_split['val_masks'], healthMasksLoader=self.get_healty_masks_transforms()),
                ClipAndZNormalizationd(keys=['mri','healthy_region_mask_corrected', 'unhealthy_mask'], intensity_scale_range=self.intensity_scale_range),
                CopyItemsd(keys=["voided_mri","healthy_region_mask_corrected"], times=1, names=["voided_mri_full_res","healthy_region_mask_corrected_full_res"]),
                RandPatchByMaskd(
                    keys=["voided_mri","healthy_region_mask_corrected","target"],
                    spatial_size=self.spatial_size,
                    num_patches=1,
                    min_mask_voxels=4,
                ),
                ToTensord(keys=["voided_mri","healthy_region_mask_corrected","target",'mean','std'])
            ]
        return io_transforms, random_transforms

    def get_loaders_val(self):
        io_transforms, random_transforms = self.get_io_transforms_val()


        cached_ds_mri = CacheDataset(
            data=self.data_split['val_cases_mri'],
            transform=Compose(io_transforms),
            cache_rate=self.cache_rate,   # only deterministic steps cached
            num_workers=self.num_workers,
        )

        final_ds_mri = Dataset(
            data=cached_ds_mri,
            transform=Compose(random_transforms),
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