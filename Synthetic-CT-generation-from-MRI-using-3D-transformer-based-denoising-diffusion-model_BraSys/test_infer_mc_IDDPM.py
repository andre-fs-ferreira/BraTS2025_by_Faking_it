print("Start importing...")
# Standard library
import os
import sys
import time
import json
import argparse
import random

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Modify sys.path
sys.path.append("..")
sys.path.append(".")

# Third-party libraries
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm.auto import tqdm

# MONAI core
import monai
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from monai.utils.enums import TransformBackends
from monai.data import ITKReader, CacheDataset, Dataset
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import pad_list_data_collate
from torch.utils.data import DataLoader

from monai.data.meta_obj import get_track_meta
# MONAI transforms
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged,
    RandAffined, RandGaussianNoised, RandAdjustContrastd, RandBiasFieldd,
    RandShiftIntensityd, RandScaleIntensityd, RandGridDistortiond,
    RandGaussianSmoothd, RandSpatialCropSamplesd, CropForegroundd,
    DeleteItemsd, NormalizeIntensityd, ToTensord, EnsureType,
    CopyItemsd, ResizeWithPadOrCropd, SpatialCropd,
    LoadImage, EnsureChannelFirst, Orientation, CropForeground,
    NormalizeIntensity, ClipIntensityPercentiles
)
from monai.transforms.utils_pytorch_numpy_unification import clip, percentile
from monai.transforms import MapTransform, Randomizable, Transform
from monai.transforms.transform import MapTransform as CoreMapTransform, RandomizableTransform
from typing import Hashable, Mapping, Sequence, Tuple
from collections.abc import Hashable as ABC_Hashable, Mapping as ABC_Mapping, Sequence as ABC_Sequence

# Project-specific imports
from train_mc_IDDPM_BraSyn import *
from diffusion.Create_diffusion import *
from diffusion.resampler import *
from network.Diffusion_model_transformer import *
from network.Pre_trained_networks import (
    load_pretrained_swinvit, load_pretrained_SwinUNETR,
    load_pretrained_TotalSegmentator, freeze_layers
)

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
    NormalizeIntensity, ClipIntensityPercentiles, ScaleIntensityd, ScaleIntensity
)
from monai.transforms import MapTransform, Randomizable, Transform
from monai.transforms.transform import MapTransform as CoreMapTransform, RandomizableTransform
from monai.utils import set_determinism
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype

# Path utilities
from os.path import join
import nibabel as nib
from nibabel.orientations import (
    aff2axcodes,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)

print("Finished importing.")

##############################################################
##### Functions to prepare the input and post processing #####
##############################################################

def create_data_dict(args):
    all_cases = {}
    cases_list = []
    for case_id in listdir(args.data_dir):
        case_id_path = join(args.data_dir, case_id)
        entry = {
            't1':  join(case_id_path, f"{case_id}-t1n.nii.gz"),
            't1c':  join(case_id_path, f"{case_id}-t1c.nii.gz"),
            't2':  join(case_id_path, f"{case_id}-t2w.nii.gz"), # 
            'flair':  join(case_id_path, f"{case_id}-t2f.nii.gz")
        }
        flag=False
        for case_to_check_key in entry.keys():
            if not os.path.exists(entry[case_to_check_key]):
                # TODO this file should be copied when creating the docker. Save it in a defined folder.
                entry[case_to_check_key] =  '/homes/andre.ferreira/BraTS2025/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model_BraSys/empty_tensor/zeros_output.nii.gz' 
                if flag:
                    print(f"Warning: {case_id} has more than one modality missing. Using empty_tensor for {case_to_check_key}.")
                flag=True
        flag=False
                
        cases_list.append(entry)
    if args.mode == 'test':
        all_cases['test_cases_mri'] = cases_list
    elif args.mode == 'val':
        all_cases['val_cases_mri'] = cases_list
    elif args.mode == 'train':
        all_cases['train_cases_mri'] = cases_list
    return all_cases

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

class ClipIntensityPercentilesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ClipIntensityPercentiles`.
    Clip the intensity values of input image to a specific range based on the intensity distribution of the input.
    If `sharpness_factor` is provided, the intensity values will be soft clipped according to
    f(x) = x + (1/sharpness_factor) * softplus(- c(x - minv)) - (1/sharpness_factor)*softplus(c(x - maxv))
    """

    def __init__(
        self,
        keys: KeysCollection,
        lower: float | None,
        upper: float | None,
        sharpness_factor: float | None = None,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ClipIntensityPercentiles(
            lower=lower, upper=upper, sharpness_factor=sharpness_factor, channel_wise=channel_wise, dtype=dtype
        )

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            #print(d[f"{key}_meta_dict"])
            #print("#################")
            d[key] = self.scaler(d[key])
        #print("##########################################################################################################################################################################")
        return d

class ClipIntensityPercentiles(Transform):
    """
    Apply clip based on the intensity distribution of input image.
    If `sharpness_factor` is provided, the intensity values will be soft clipped according to
    f(x) = x + (1/sharpness_factor)*softplus(- c(x - minv)) - (1/sharpness_factor)*softplus(c(x - maxv))
    From https://medium.com/life-at-hopper/clip-it-clip-it-good-1f1bf711b291

    Soft clipping preserves the order of the values and maintains the gradient everywhere.
    For example:

    .. code-block:: python
        :emphasize-lines: 11, 22

        image = torch.Tensor(
            [[[1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5]]])

        # Hard clipping from lower and upper image intensity percentiles
        hard_clipper = ClipIntensityPercentiles(30, 70)
        print(hard_clipper(image))
        metatensor([[[2., 2., 3., 4., 4.],
                [2., 2., 3., 4., 4.],
                [2., 2., 3., 4., 4.],
                [2., 2., 3., 4., 4.],
                [2., 2., 3., 4., 4.],
                [2., 2., 3., 4., 4.]]])


        # Soft clipping from lower and upper image intensity percentiles
        soft_clipper = ClipIntensityPercentiles(30, 70, 10.)
        print(soft_clipper(image))
        metatensor([[[2.0000, 2.0693, 3.0000, 3.9307, 4.0000],
         [2.0000, 2.0693, 3.0000, 3.9307, 4.0000],
         [2.0000, 2.0693, 3.0000, 3.9307, 4.0000],
         [2.0000, 2.0693, 3.0000, 3.9307, 4.0000],
         [2.0000, 2.0693, 3.0000, 3.9307, 4.0000],
         [2.0000, 2.0693, 3.0000, 3.9307, 4.0000]]])

    See Also:

        - :py:class:`monai.transforms.ScaleIntensityRangePercentiles`
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        lower: float | None,
        upper: float | None,
        sharpness_factor: float | None = None,
        channel_wise: bool = False,
        return_clipping_values: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        """
        Args:
            lower: lower intensity percentile. In the case of hard clipping, None will have the same effect as 0 by
                not clipping the lowest input values. However, in the case of soft clipping, None and zero will have
                two different effects: None will not apply clipping to low values, whereas zero will still transform
                the lower values according to the soft clipping transformation. Please check for more details:
                https://medium.com/life-at-hopper/clip-it-clip-it-good-1f1bf711b291.
            upper: upper intensity percentile.  The same as for lower, but this time with the highest values. If we
                are looking to perform soft clipping, if None then there will be no effect on this side whereas if set
                to 100, the values will be passed via the corresponding clipping equation.
            sharpness_factor: if not None, the intensity values will be soft clipped according to
                f(x) = x + (1/sharpness_factor)*softplus(- c(x - minv)) - (1/sharpness_factor)*softplus(c(x - maxv)).
                defaults to None.
            channel_wise: if True, compute intensity percentile and normalize every channel separately.
                default to False.
            return_clipping_values: whether to return the calculated percentiles in tensor meta information.
                If soft clipping and requested percentile is None, return None as the corresponding clipping
                values in meta information. Clipping values are stored in a list with each element corresponding
                to a channel if channel_wise is set to True. defaults to False.
            dtype: output data type, if None, same as input image. defaults to float32.
        """
        if lower is None and upper is None:
            raise ValueError("lower or upper percentiles must be provided")
        if lower is not None and (lower < 0.0 or lower > 100.0):
            raise ValueError("Percentiles must be in the range [0, 100]")
        if upper is not None and (upper < 0.0 or upper > 100.0):
            raise ValueError("Percentiles must be in the range [0, 100]")
        if upper is not None and lower is not None and upper < lower:
            raise ValueError("upper must be greater than or equal to lower")
        if sharpness_factor is not None and sharpness_factor <= 0:
            raise ValueError("sharpness_factor must be greater than 0")

        self.lower = lower
        self.upper = upper
        self.sharpness_factor = sharpness_factor
        self.channel_wise = channel_wise
        if return_clipping_values:
            self.clipping_values: list[tuple[float | None, float | None]] = []
        self.return_clipping_values = return_clipping_values
        self.dtype = dtype

    def _clip(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if self.sharpness_factor is not None:
            lower_percentile = percentile(img, self.lower) if self.lower is not None else None
            upper_percentile = percentile(img, self.upper) if self.upper is not None else None
            img = soft_clip(img, self.sharpness_factor, lower_percentile, upper_percentile, self.dtype)
        else:
            lower_percentile = percentile(img, self.lower) if self.lower is not None else percentile(img, 0)
            upper_percentile = percentile(img, self.upper) if self.upper is not None else percentile(img, 100)
            img = clip(img, lower_percentile, upper_percentile)

        if self.return_clipping_values:
            self.clipping_values.append(
                (
                    (
                        lower_percentile
                        if lower_percentile is None
                        else lower_percentile.item() if hasattr(lower_percentile, "item") else lower_percentile
                    ),
                    (
                        upper_percentile
                        if upper_percentile is None
                        else upper_percentile.item() if hasattr(upper_percentile, "item") else upper_percentile
                    ),
                )
            )
        img = convert_to_tensor(img, track_meta=False)
        return img

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t = convert_to_tensor(img, track_meta=False)
        if self.channel_wise:
            img_t = torch.stack([self._clip(img=d) for d in img_t])  # type: ignore
        else:
            img_t = self._clip(img=img_t)

        img = convert_to_dst_type(img_t, dst=img)[0]
        if self.return_clipping_values:
            img.meta["clipping_values"] = self.clipping_values  # type: ignore

        return img

class MonaiDataLoader():
    def __init__(self,
        spatial_size,
        use_global_stats,
        batch_size=1,
        num_workers=4,
        json_file='/homes/andre.ferreira/BraTS2025/Dataset/Global-Synthesis_Train/data_split.json',
    ):
        self.spatial_size = spatial_size
        self.batch_size = batch_size
        self.use_global_stats = use_global_stats
        self.num_workers=num_workers
        if use_global_stats:
            with open(json_file.replace('data_split.json', 'stats.json'), 'r') as file:
                self.stats_data = json.load(file)
    
    def get_io_transforms(self, dtype: torch.dtype = torch.float32):
        keys_list = ['t1', 't1c', 't2', 'flair']  

        # Load data to memory transforms
        io_transforms = [
            LoadImaged(keys=keys_list, image_only=False, reader=ITKReader()),
            EnsureChannelFirstd(keys=keys_list),
            Orientationd(keys=keys_list, axcodes="RAS"),
            CreateForegroundMaskd(keys=keys_list),   
            CropForegroundd(keys=keys_list+['foreground_mask'], source_key='foreground_mask'),
            ToTensord(keys=keys_list+['foreground_mask'])
        ]
        return io_transforms
      
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
            ToTensord(keys=keys_list+['foreground_mask'])
            )
                                    
        return out_transforms

    def get_loaders_val(self):
        io_transforms = self.get_io_transforms()
        out_transforms = self.get_out_transforms_val()

        if args.mode == 'test':
            json_key = 'test_cases_mri'
            self.data_split = create_data_dict(args)
        elif args.mode == 'val':
            json_key = 'val_cases_mri'
            self.data_split = create_data_dict(args)
        elif args.mode == 'train':
            json_key = 'train_cases_mri'
            self.data_split = create_data_dict(args)
            
        print(f"Predicting: {json_key}")
        print(f"Number of cases {len(self.data_split[json_key])}")
        ds_mri = Dataset(
            data=self.data_split[json_key],
            transform=Compose(io_transforms),
        )

        final_ds_mri = Dataset(
            data=ds_mri,
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
        )
        return val_loader_mri

def get_diffusion(timestep_respacing, timestep_respacing_val, args):
    """
    Define the gaussian diffusion scheduler for training.
    These three parameters: training steps number, learning variance or not (using improved DDPM or original DDPM), and inference 
    timesteps number (only effective when using improved DDPM)
    In:
        timestep_respacing: Used mainly for inference to reduce the number of steps.
        timestep_respacing_val: Used mainly for inference to reduce the number of steps.
    Out:
        train_diffusion, val_diffusion, schedule_sampler: Diffusion models and the schedule sampler
        # val_diffusion has less time steps for inference
    """
    # Hard coded parameters
    #  
    sigma_small=False
    noise_schedule='linear'
    use_kl=False
    predict_xstart=True
    rescale_timesteps=True
    rescale_learned_sigmas=True
    diffusion_steps=1000
    learn_sigma=True

    test_diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing_val,
    )

    return test_diffusion

def get_model(device, args):
    """
    Initializes and returns a model based on the specified network type.

    Args:
        device (torch.device): The device on which the model will be loaded 
            (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The initialized model corresponding to the specified 
            network type, along with filtered_dict and not_matching_keys 
            (if applicable).

    Raises:
        ValueError: If the specified network type is not recognized. Valid 
            options are 'SwinVIT', 'SwinUNETR_vit', 'SwinUNETR', or 'nnUNet'.

    Note:
        The network type is determined by the global `args.network` variable.
        Ensure that `args.network` is set to one of the valid options before 
        calling this function.
    """
    filtered_dict, not_matching_keys = None, None
    if args.network=="SwinVIT":
        A_to_B_model = get_SwinVITmodel(args)

    elif args.network=="SwinUNETR_vit":
        A_to_B_model, filtered_dict, not_matching_keys = load_pretrained_swinvit(
                                                            ckpt_path=args.path_pretrained, 
                                                            load_weights=args.load_pretrained, 
                                                            img_size=args.patch_size, 
                                                            in_channels=2, 
                                                            out_channels=2, 
                                                            feature_size=48, 
                                                            use_checkpoint=True, 
                                                            verbose=args.verbose
                                                            )
    elif args.network=="SwinUNETR":
        A_to_B_model, filtered_dict, not_matching_keys = load_pretrained_SwinUNETR(
                                                            ckpt_path=args.path_pretrained, 
                                                            load_weights=args.load_pretrained, 
                                                            img_size=args.patch_size, 
                                                            in_channels=2, 
                                                            out_channels=2, 
                                                            feature_size=48, 
                                                            use_checkpoint=True, 
                                                            verbose=args.verbose
                                                            )
    elif args.network=="nnUNet":
        A_to_B_model, filtered_dict, not_matching_keys = load_pretrained_TotalSegmentator(
                                                            ckpt_path=args.path_pretrained, 
                                                            load_weights=args.load_pretrained,
                                                            verbose=args.verbose
                                                            )
    else:
        raise ValueError("Unknown network type. " \
        "Please choose 'SwinVIT' (for original implementation),"
        " 'SwinUNETR_vit' (for loading vit pre trained weights),"
        " 'SwinUNETR' (to train from scratch or loading pre-trained weights from SwinUNETR),"
        " or 'nnUNet' (to train from scratch or loading pre-trained weights from TotalSegmentator).")

    A_to_B_model = A_to_B_model.to(device)
    return A_to_B_model, filtered_dict, not_matching_keys

def get_nifti_metadata(file_path):
    """
    Extract the header and affine matrix from a NIfTI (.nii or .nii.gz) file.

    Parameters:
        file_path (str): Path to the .nii or .nii.gz file

    Returns:
        tuple: (header, affine)
            - header: Nibabel header object (contains metadata)
            - affine: numpy.ndarray (4x4 transformation matrix)
    """
    img = nib.load(file_path)
    header = img.header
    affine = img.affine
    return header, affine

def diffusion_sampling(condition, model):
    if 'ddim' in args.timestep_respacing_val:
        print(f"Doing ddim")
        sampling_method = test_diffusion.ddim_sample_loop
    else:
        print(f"Doing p_sampling")
        sampling_method = test_diffusion.p_sample_loop

    sampled_images = sampling_method(
            model,
            (condition.shape[0], 1, condition.shape[2], condition.shape[3],condition.shape[4]),
            condition=condition,
            clip_denoised=args.clip_denoised
            )
    if torch.isnan(sampled_images).any(): 
        print(f"NaN detected in the sampled_images in inference. Returning the condition instead.")
        
    return sampled_images

def ras_to_target_orient(data, target_affine):
    """
    Re‑orient data from RAS → the orientation encoded by target_affine.
    
    Inputs:
      - data: np.ndarray or torch.Tensor, shape (H,W,D) or (C,H,W,D)
      - target_affine: array-like (4×4), the affine whose orientation you want to match
    
    Returns:
      - reoriented data (same type as input)
      - (you keep using target_affine as-is)
    """
    # 1) Get target axis codes, e.g. ('L','P','S')
    aff_np = np.asarray(target_affine).squeeze()
    target_codes = aff2axcodes(aff_np)
    print("Target orientation codes:", target_codes)
    if tuple(target_codes) == ('R','A','S'):
        print("✅ Target is RAS – no change.")
        return data

    # 2) Build orientation transforms
    ras_ornt    = axcodes2ornt(('R','A','S'))
    target_ornt = axcodes2ornt(target_codes)
    xfm         = ornt_transform(ras_ornt, target_ornt)

    # 3) Apply to data
    is_torch = isinstance(data, torch.Tensor)
    arr = data.cpu().numpy() if is_torch else np.asarray(data)
    arr_oriented = apply_orientation(arr, xfm)

    print(f"Re‑oriented data: RAS → {target_codes}")
    return torch.from_numpy(arr_oriented) if is_torch else arr_oriented

def evaluate(model, test_loader_mri, inferer, args):
    """
    The result will be saved in nii.gz format
    """
    scale_intensity = ScaleIntensity(
        minv=0.0, 
        maxv=1.0, 
        factor=None, 
        channel_wise=False
    )
    ending_name = args.path_checkpoint.split('_')[-1].replace('.pt','')
    if args.clip_denoised:
        ending_name = f"{ending_name}_clip"
    else:
        ending_name = f"{ending_name}_"

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader_mri, desc="Inferece")):
            print(f"Doing case {i} from {args.start_case} to {args.end_case}")
            if i >= int(args.start_case)  and i < int(args.end_case):
                # Define the padding to save the files
                foreground_start = batch['foreground_start_coord'][0] # This assumes batch size 1!
                foreground_end = batch['foreground_end_coord'][0] # This assumes batch size 1!
                target_shape = torch.tensor([240, 240, 155])
                post_pad = target_shape - foreground_end
                padding = [
                    foreground_start[2].item(), post_pad[2].item(),
                    foreground_start[1].item(), post_pad[1].item(), 
                    foreground_start[0].item(), post_pad[0].item(),
                ]
    
                # Define condition
                t1 = batch["t1"].to(args.device)
                t1c = batch["t1c"].to(args.device)
                t2 = batch["t2"].to(args.device)
                flair = batch["flair"].to(args.device)

                condition = torch.cat([t1, t1c, t2, flair], dim=1)
                condition = condition.to(args.device)
                cond_keys = ['t1', 't1c', 't2', 'flair']

                # Find what modality is missing:
                for i, modality in enumerate(condition[0]):  # condition[0] has shape (4, H, W, ...)
                    if batch[f'{cond_keys[i]}_meta_dict']['filename_or_obj'][0].split('/')[-2]=='empty_tensor':
                        print(f"Modality '{cond_keys[i]}' is missing.") 
                        missing_modal_key = f"{cond_keys[i]}-pred" 
                        condition[0][i] = torch.ones_like(condition[0][i])
                    else:
                        affine_original = batch[f'{cond_keys[i]}_meta_dict']['original_affine']
                
                # Getting std and mean for missing modal
                args.global_std=stats_data[missing_modal_key.replace('-pred','')]['std']
                args.global_mean=stats_data[missing_modal_key.replace('-pred','')]['mean']

  
                if missing_modal_key!='t1-pred':
                    mri_file_path = batch['t1_meta_dict']['filename_or_obj'][0]
                    patient_id = batch['t1_meta_dict']['filename_or_obj'][0].split('/')[-2]
                    header, affine = get_nifti_metadata(mri_file_path) 
                else:
                    mri_file_path = batch['t1c_meta_dict']['filename_or_obj'][0]
                    patient_id = batch['t1c_meta_dict']['filename_or_obj'][0].split('/')[-2]
                    header, affine = get_nifti_metadata(mri_file_path) 
                exp_name = args.path_checkpoint.split('/')[-6]

                assert patient_id!='empty_tensor' 
                os.makedirs(f'{args.prediction_path}/{exp_name}/{args.mode}/{args.exp_name}/{args.timestep_respacing_val}{ending_name}/{patient_id}', exist_ok=True)

                for cond_case, cond_key in zip(condition[0], cond_keys):
                    padded_case = F.pad(cond_case, padding, mode='constant', value=-1)
                    padded_case = padded_case.cpu().numpy()
                    print(f"padded_case.shape: {padded_case.shape}")
                    padded_case = ras_to_target_orient(padded_case, affine_original)
                    norm_case = scale_intensity(padded_case)
                    #if 'BraTS-GLI' in patient_id:
                    #padded_case = np.flip(padded_case, axis=(0,1))
                    # save mask
                    nib.save(
                        nib.Nifti1Image(norm_case, affine=affine, header=header), 
                        f'{args.prediction_path}/{exp_name}/{args.mode}/{args.exp_name}/{args.timestep_respacing_val}{ending_name}/{patient_id}/{patient_id}-{cond_key}.nii.gz'
                        )

        
                print(f"condition: {condition.shape}")
                #original_mean, original_std = batch['mean'], batch['std']
                output_file = f'{args.prediction_path}/{exp_name}/{args.mode}/{args.exp_name}/{args.timestep_respacing_val}{ending_name}/{patient_id}/{patient_id}-{missing_modal_key}.nii.gz'

                if os.path.exists(output_file):
                    print(f"{output_file} exists already!")
                    continue

                # Prediction
                # condition is the input voided MRI        
                # sampled_images is the synthetic CT
                with torch.amp.autocast("cuda"):
                    sampled_images = inferer(
                        condition,
                        diffusion_sampling,
                        model
                        )
                    if torch.isnan(sampled_images).any(): 
                        print(f"NaN detected in the sampled_images in inference.")
                    
                # Save prediction
                if len(sampled_images.shape) == 5:
                    sampled_images = sampled_images[0][0]
                elif len(sampled_images.shape) == 4:
                    sampled_images = sampled_images[0]     
                sampled_images = F.pad(sampled_images, padding, mode='constant', value=-1)
                nii_pred_image = sampled_images.cpu().numpy() # Identity matrix as the affine transformation
                if args.use_global_stats:
                    nii_pred_image = nii_pred_image * args.global_std + args.global_mean 
    

                #if 'BraTS-GLI' in patient_id:
                #    nii_pred_image = np.flip(nii_pred_image, axis=(0,1))
                # TODO remove background and normalize between -1 and 1
                nii_pred_image = ras_to_target_orient(nii_pred_image, affine_original)
                nii_pred_image = scale_intensity(nii_pred_image)
                nib.save(
                    nib.Nifti1Image(nii_pred_image, affine=affine, header=header), 
                    output_file
                    )

def get_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--network', type=str, default='SwinVIT', help='Network type: SwinVIT (default), SwinUNETR_vit (for pre-trained vit), SwinUNETR (scratch or pre-trained SwinUNETR), nnUNet (scratch or pre-trained TotalSegmentator)')
    parser.add_argument('--batch_size_train', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 64], help='Patch size in (x, y, z)')
    parser.add_argument('--patch_num', type=int, default=2, help='Number of patches extracted per volume')
    parser.add_argument('--dataset_path', type=str, default='../../Dataset', help='Path to the dataset')
    parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate for MONAI DataLoader CacheDataset')
    parser.add_argument('--train_metric', type=str, choices=['MAE', 'MSE'], default='MSE', help='Train metric choice: MAE (L1) or MSE (L2)')
    parser.add_argument('--timestep_respacing', type=str, help='Timestep respacing values, e.g., 50 100')
    parser.add_argument('--timestep_respacing_val', type=str, help='Timestep respacing values for validation, e.g., ddim50')
    parser.add_argument('--verbose', action='store_true', help='Verbose output for detailed information')
    parser.add_argument('--shuffle', action='store_true', help='Use shuffle in the data loader')
    parser.add_argument('--sw_batch_size', type=int, default=12, help='Sliding window batch size')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap for sliding window')
    parser.add_argument('--overlap_mode', type=str, default='constant', help='Overlap mode for sliding window. constant or gaussian')
    parser.add_argument('--path_checkpoint', type=str, default='../../results/', help='Path to the saved model checkpoint')
    parser.add_argument('--freeze_epochs', type=int, default=0, help='For how many epochs the loaded weights should be frozen')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from, e.g., args.path_checkpoint/A_to_B_ViTRes1_latest.pt')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--pacience', type=int, default=10, help='Number of epochs to wait before stoppin the training if val does not improve.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout to apply to the model.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to start with.')
    parser.add_argument('--val_interval', type=int, default=50, help='Validation interval (in epochs)')
    parser.add_argument('--prob', type=float, default=0.0, help='Probability of applying transforms (data augmentation aggressiveness)')
    parser.add_argument('--add_train_metric', nargs='+', type=str, default=[], help='List of metrics to compute loss. Default is an empty list.' )
    parser.add_argument('--add_train_metric_weight', nargs='+', type=float, default=[], help='List of weights for each add_train_metric metric. Default is an empty list.' )
    parser.add_argument('--intensity_scale_range', action='store_true', help='Linearly normalize the intensities to -1 and 1 (from the z-score).')
    parser.add_argument('--json_file', type=str, default='/homes/andre.ferreira/BraTS2025/Dataset/Global-Synthesis_Val/data_split.json', help='Json file for the data split.')
    parser.add_argument('--mode', type=str, default='test', help='Key for the datasplit, train, val, test. Default test.')
    parser.add_argument('--prediction_path', type=str, default='test', help='Key for the datasplit, train, val, test. Default test.')
    parser.add_argument('--use_global_stats', action='store_true', help='Linearly normalize the intensities to -1 and 1 (from the z-score).')
    parser.add_argument('--data_dir', type=str, help='Root directory with the subdirectories with the cases to predict.')
    parser.add_argument('--start_case', type=str, default='0', help='Case number to start inference.')
    parser.add_argument('--end_case', type=str, default='1000', help='Case number to finish inference.')
    parser.add_argument('--clip_denoised', action='store_true', help='If clip at every inference.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    seed_value = 42
    set_complete_seed(seed_value)
    args = get_args()
    torch.backends.cudnn.benchmark = True

    #######################
    ###### Arguments ######
    #######################
    seed_value = 42
    set_complete_seed(seed_value)
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.exp_name = args.path_checkpoint.split('logs/')[-1].split('/wandb')[0]
    if args.verbose:
        print("########### Arguments used ###########")
        print(f"Batch Size (Train): {args.batch_size_train}")
        print(f"Number of Workers: {args.num_workers}")
        print(f"Patch Size: {args.patch_size}")
        print(f"patch_num Size: {args.patch_num}")
        print(f"Dataset Path: {args.dataset_path}")
        print(f"Cache Rate: {args.cache_rate}")
        print(f"Timestep Respacing: {args.timestep_respacing}")
        print(f"Timestep Respacing Validation: {args.timestep_respacing_val}")
        print(f"Sliding Window Batch Size: {args.sw_batch_size}")
        print(f"Overlap: {args.overlap}")
        print(f"Overlap mode: {args.overlap_mode}")
        print(f"Checkpoint Path: {args.path_checkpoint}")
        print(f"Resume Training: {args.resume}")
        print(f"Number of Epochs: {args.n_epochs}")
        print(f"Validation Interval: {args.val_interval} epochs")
        print(f"Using add_train_metric: {args.add_train_metric}")
        print(f"Using add_train_metric_weight: {args.add_train_metric_weight}")
        print(f"Using dropout: {args.dropout}")
        print(f"Using intensity ranged: {args.intensity_scale_range}")
        print(f"Using device: {args.device}")
        print(f"Using global stats for z-score: {args.use_global_stats}")
        print(f"Clip denoised: {args.clip_denoised}")
        print("####################################")

    ########################################
    ######### Define data loader ##########
    ########################################
    dataLoader_obj = MonaiDataLoader(
            spatial_size=args.patch_size,
            use_global_stats=args.use_global_stats,
            batch_size=1,
            num_workers=args.num_workers,
            json_file=args.json_file,
        )
    test_loader_mri  = dataLoader_obj.get_loaders_val()
    if args.use_global_stats:
        with open(args.json_file.replace('data_split.json', 'stats.json'), 'r') as file:
            stats_data = json.load(file)

    ############################################
    ######### Get model and diffusion ##########
    ############################################
    test_diffusion = get_diffusion(
        timestep_respacing=args.timestep_respacing,
        timestep_respacing_val=args.timestep_respacing_val,
        args=args
        )
    A_to_B_model, filtered_dict, not_matching_keys = get_model(
        device=args.device,
        args=args
        )
    checkpoint = torch.load(join(args.path_checkpoint), weights_only=False)
    A_to_B_model.load_state_dict(checkpoint['model_state_dict'])
    A_to_B_model.eval()
    A_to_B_model.cuda()

    
    inferer = SlidingWindowInferer(
        roi_size=(args.patch_size[0], args.patch_size[1], args.patch_size[2]), 
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap, 
        mode=args.overlap_mode, 
        progress=True
        )
    evaluate(
        model=A_to_B_model, 
        test_loader_mri=test_loader_mri, 
        inferer=inferer, 
        args=args)
