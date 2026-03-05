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

# MONAI transforms
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
from typing import Hashable, Mapping, Sequence, Tuple
from collections.abc import Hashable as ABC_Hashable, Mapping as ABC_Mapping, Sequence as ABC_Sequence

# Project-specific imports
from train_mc_IDDPM import *
from diffusion.Create_diffusion import *
from diffusion.resampler import *
from network.Diffusion_model_transformer import *
from network.Pre_trained_networks import (
    load_pretrained_swinvit, load_pretrained_SwinUNETR,
    load_pretrained_TotalSegmentator, freeze_layers
)

# Path utilities
from os.path import join



print("Finished importing.")

##############################################################
##### Functions to prepare the input and post processing #####
##############################################################
class ClipAndZNormalizationd(MapTransform):
    """
    Compute voided and target volumes, then clip + z‑normalize both.

    voided = mri * (1 - corrected_mask) * (1 - unhealthy_mask)
    target = mri * corrected_mask

    Args:
        keys: sequence of three keys:
            [mri_key, corrected_mask_key, unhealthy_mask_key]
        voided_key: name under which to store the voided output
        lower: lower percentile for clipping (default 0.1)
        upper: upper percentile for clipping (default 99.9)
        allow_missing_keys: if True, skip missing keys without error
    """
    def __init__(
        self,
        keys: KeysCollection,
        lower: float = 0.1,
        upper: float = 99.9,
        intensity_scale_range: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.clipper = ClipIntensityPercentiles(
            lower=lower, upper=upper,
            sharpness_factor=None,
            channel_wise=False,
            return_clipping_values=False,
        )
        self.intensity_scale_range = intensity_scale_range

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        mri_voided_key = self.keys[0]
        mri_voided = d[mri_voided_key]  

        # clip intensities
        mri_voided = self.clipper(mri_voided)

        # compute stats on nonzero of voided
        mask_foreground = mri_voided != 0
        vals = mri_voided[mask_foreground]
        mean, std = float(vals.mean()), float(vals.std())

        # normalize both
        normalizer = NormalizeIntensity(
            subtrahend=mean,
            divisor=std,
            nonzero=False,
            channel_wise=False,
        )
        mri_voided = normalizer(mri_voided)
        # save max and min in case we do intensity_scale_range
        d['z_score_max'] = mri_voided.max()
        d['z_score_min'] = mri_voided.min()

        if self.intensity_scale_range:
            intensity_scaler = ScaleIntensity(
                minv=-1.0,
                maxv=1.0,
                )
            mri_voided = intensity_scaler(mri_voided)
                                   
        # write back
        d[mri_voided_key] = mri_voided
        d['mean'] = mean
        d['std'] = std
        
        return d

class MonaiDataLoader():
    def __init__(self,
        spatial_size,
        num_workers,
        intensity_scale_range=False,
        json_file='/homes/andre.ferreira/BraTS2025/Dataset/Local-Synthesis/data_split.json',
    ):
        self.spatial_size = spatial_size
        self.num_workers = num_workers
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

    def get_io_transforms_test(self, dtype: torch.dtype = torch.float32):
        keys_list = ['mri', 'unhealthy_mask'] 

        # Load data to memory transforms
        io_transforms = [
            LoadImaged(keys=keys_list, image_only=False, reader=ITKReader()),
            EnsureChannelFirstd(keys=keys_list),
            Orientationd(keys=keys_list, axcodes="RAS"),
            ClipAndZNormalizationd(keys=['mri'], intensity_scale_range=self.intensity_scale_range),
            ToTensord(keys=["mri","unhealthy_mask",'mean','std'])
            ]
        return io_transforms

    def get_loaders_test(self):
        io_transforms = self.get_io_transforms_test()

        if args.mode == 'test':
            json_key = 'test_cases_mri'
        elif args.mode == 'val':
            json_key = 'val_cases_mri'
        elif args.mode == 'train':
            json_key = 'train_cases_mri'
        print(f"Predicting: {json_key}")
        ds_mri = Dataset(
            data=self.data_split[json_key],
            transform=Compose(io_transforms),
        )

        test_loader_mri = DataLoader(
                ds_mri,
                batch_size=1,
                shuffle=False,                 # let DataLoader shuffle
                generator=torch.Generator(),  # we’ll reseed it each epoch
                num_workers=self.num_workers,
                persistent_workers=True,      # keep workers alive across epochs
                pin_memory=True,
                prefetch_factor=2,
                collate_fn=pad_list_data_collate
        )
        return test_loader_mri

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
    noise_schedule=args.noise_schedule
    use_kl=False
    predict_xstart=True
    rescale_timesteps=True
    rescale_learned_sigmas=True
    diffusion_steps=1000
    learn_sigma=True

    if 'inpaint' in timestep_respacing_val:
        timestep_respacing_val = timestep_respacing_val.replace('inpaint', '')
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
    if torch.sum(condition[0][1])==0:
        return condition[:1, :1 , :, :, :]
    
    else:
        if 'ddim' in args.timestep_respacing_val:
            if 'inpaint' in args.timestep_respacing_val:
                print(f"Doing ddim inpaint")
                sampling_method = test_diffusion.ddim_sample_loop_inpaint
            else:
                print(f"Doing ddim")
                sampling_method = test_diffusion.ddim_sample_loop
        else:
            if 'inpaint' in args.timestep_respacing_val:
                print(f"Doing p_sampling inpaint")
                sampling_method = test_diffusion.p_sample_loop_inpaint
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
            return condition[:1, :1 , :, :, :]
        return sampled_images

def linear_denormalize(data, new_min, new_max):
    """
    Normalize data from [-1, 1] range to a new range [new_min, new_max].

    Parameters:
        data (np.ndarray): Input array with values in the range [-1, 1].
        new_min (float): Desired minimum value of the output.
        new_max (float): Desired maximum value of the output.

    Returns:
        np.ndarray: Normalized array in the range [new_min, new_max].
    """
    #if not np.all((data >= -1) & (data <= 1)):
    #    raise ValueError("Input data must be in the range [-1, 1]")

    # Normalize to [0, 1]
    data_01 = (data + 1) / 2.0

    # Scale to [new_min, new_max]
    return new_min + data_01 * (new_max - new_min)

def evaluate(model, test_loader_mri, inferer, args):
    """
    The result will be saved in nii.gz format
    """
    model.eval()

    ending_name = args.path_checkpoint.split('_')[-1].replace('.pt','')
    if args.clip_denoised:
        ending_name = f"{ending_name}_clip"
    else:
        ending_name = ending_name
    os.makedirs(f'{args.prediction_path}/{args.mode}/{args.exp_name}/{args.timestep_respacing_val}{ending_name}', exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader_mri, desc="Inferece")):
            print(f"Doing case {i} from {args.start_case} to {args.end_case}")
            if i >= int(args.start_case)  and i <= int(args.end_case):
                voided_mri = batch["mri"].reshape(-1, 1, 240, 240, 155)
                unhealthy_mask = batch["unhealthy_mask"].reshape(-1, 1, 240, 240, 155)

                # Get metadata
                mri_file_path = batch['mri_meta_dict']['filename_or_obj'][0]
                patient_id = batch['mri_meta_dict']['filename_or_obj'][0].split('/')[-2]
                header, affine = get_nifti_metadata(mri_file_path) 

                # save mask
                mask_numpy = unhealthy_mask[0][0].cpu().numpy()
                #mask_numpy = mask_numpy.swapaxes(0, 1) 
                mask_numpy = np.flip(mask_numpy, axis=(0,1))
                nib.save(
                    nib.Nifti1Image(mask_numpy, affine=affine, header=header), 
                    f'{args.prediction_path}/{args.mode}/{args.exp_name}/{args.timestep_respacing_val}{ending_name}/{patient_id}-mask.nii.gz'
                    )

                print(f"unhealthy_mask: {unhealthy_mask.shape}")
                print(f"voided_mri: {voided_mri.shape}")
                original_mean, original_std = batch['mean'], batch['std']
                output_file = f'{args.prediction_path}/{args.mode}/{args.exp_name}/{args.timestep_respacing_val}{ending_name}/{patient_id}-t1n-inference.nii.gz'
                
                if os.path.exists(output_file):
                    print(f"{output_file} exists already!")
                    continue

                condition = torch.cat(
                    (voided_mri, unhealthy_mask),
                    dim=1
                ).to(args.device)

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

                predict_ct_image_batch = sampled_images.cpu().numpy() # Identity matrix as the affine transformation

                if args.intensity_scale_range:
                    z_score_max = batch['z_score_max']
                    z_score_min = batch['z_score_min']
                    predict_ct_image_batch = linear_denormalize(predict_ct_image_batch, z_score_min, z_score_max)

                nii_pred_image = predict_ct_image_batch * original_std + original_mean
                nii_pred_image[nii_pred_image < 5] = 0 # Background migh have values below 5, so we set them to 0

                nii_pred_image = np.flip(nii_pred_image, axis=(0,1))
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
    parser.add_argument('--json_file', type=str, default='/homes/andre.ferreira/BraTS2025/Dataset/Local-Synthesis/data_split.json', help='Json file for the data split.')
    parser.add_argument('--mode', type=str, default='test', help='Key for the datasplit, train, val, test. Default test.')
    parser.add_argument('--prediction_path', type=str, default='test', help='Key for the datasplit, train, val, test. Default test.')
    parser.add_argument('--start_case', type=str, default='0', help='Case number to start inference.')
    parser.add_argument('--end_case', type=str, default='1000', help='Case number to finish inference.')
    parser.add_argument('--noise_schedule', type=str, default="linear", help='Noise scheduler. linear (default) or cosine')
    parser.add_argument('--clip_denoised', action='store_true', help='If clip at every inference step.')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed_value = 42
    set_complete_seed(seed_value)
    torch.backends.cudnn.benchmark = True

    #######################
    ###### Arguments ######
    #######################
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
        print(f"Clip denoised: {args.clip_denoised}")
        print("####################################")

    ########################################
    ######### Define data loader ##########
    ########################################
    dataLoader_obj = MonaiDataLoader(
            spatial_size=args.patch_size,
            num_workers=args.num_workers,
            intensity_scale_range=args.intensity_scale_range,
            json_file=args.json_file,
        )
    
    test_loader_mri  = dataLoader_obj.get_loaders_test()

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
    
    print(f"Model from epoch: {checkpoint['epoch']}")
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
