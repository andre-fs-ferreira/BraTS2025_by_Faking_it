print("Start importing...")
import wandb
import os
import argparse
import random
import monai
import json
import sys
sys.path.append("..")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from os.path import join
from os import listdir
from os.path import isdir
from itertools import islice
import time
import torch
from diffusion.Create_diffusion import *
from diffusion.resampler import *
import torch.nn.functional as F
from monai.metrics import PSNRMetric
from monai.losses.ssim_loss import SSIMLoss
import nibabel as nib
from tqdm.auto import tqdm
from network.Diffusion_model_transformer import *
from network.Pre_trained_networks import load_pretrained_swinvit, load_pretrained_SwinUNETR, load_pretrained_TotalSegmentator, freeze_layers
from monai.inferers import SlidingWindowInferer
from monai.transforms import CropForeground
import SimpleITK as sitk
import re
from monai.metrics import PSNRMetric
from monai.metrics.regression import SSIMMetric
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from EMASmoother import EMASmoother
from MonaiDataLoader import MonaiDataLoader

print("Finished importing.")

def set_complete_seed(seed):
    """
    Sets the seed for reproducibility across multiple libraries and environments.

    Args:
        seed (int): The seed value to be set.
    """
    # Python random module
    random.seed(seed)

    # Numpy random generator
    np.random.seed(seed)

    # PyTorch random generator for CPU
    torch.manual_seed(seed)

    # PyTorch random generator for all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in PyTorch operations on CUDA
        # I will swap to make the training faster
        torch.backends.cudnn.deterministic = False # True for deterministic
        torch.backends.cudnn.benchmark = True # False for deterministic

    # MONAI deterministic behavior
    #monai.utils.set_determinism(seed=seed) # Uncomment for deterministic

    # Set environment variables (may help in some cases)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # For newer CUDA versions # Uncomment for deterministic

    print(f"Complete seed set to {seed} for reproducibility across libraries and environment.")

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


    train_diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )

    val_diffusion = create_gaussian_diffusion(
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

    schedule_sampler = UniformSampler(train_diffusion)
    return train_diffusion, val_diffusion, schedule_sampler

def get_SwinVITmodel(args):
    """
    Build the MC-IDDPM network
    Here enter your network parameters:num_channels means the initial channels in each block,
    channel_mult means the multipliers of the channels (in this case, 128,128,256,256,512,512 for the first to the sixth block),
    attention_resolution means we use the transformer blocks in the third to the sixth block
    number of heads, window size in each transformer block
    """
    # Hard coded arguments
    num_channels=64
    attention_resolutions="32,16,8"
    channel_mult = (1, 2, 3, 4)
    num_heads=[4,4,8,16]
    window_size = [[4,4,4],[4,4,4],[4,4,2],[4,4,2]]
    num_res_blocks = [2,2,2,2]
    sample_kernel=([2,2,2],[2,2,1],[2,2,1],[2,2,1]),
    use_scale_shift_norm = True
    resblock_updown = False
    dropout = args.dropout
    use_checkpoint=False

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(int(res))
    
    A_to_B_model = SwinVITModel(
            image_size=args.patch_size,
            in_channels=5, # noise + 4 modal 
            model_channels=num_channels,
            out_channels=2,
            dims=3,
            sample_kernel = sample_kernel,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=None,
            use_checkpoint=use_checkpoint,
            use_fp16=False,
            num_heads=num_heads,
            window_size = window_size,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=False,
        )
    return A_to_B_model

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

def get_additional_losses(target, model_output, args):
    """
    Calculates weighted loss values using multiple metrics.

    For each metric listed in args.add_train_metric, this function applies the corresponding
    loss function from args.more_train_metric to the model output and target, multiplies it 
    by the given weight from args.add_train_metric_weight, and stores the result.

    Parameters:
        target: Ground truth values.
        model_output: Model predictions.
        args: An object with:
            - add_train_metric: List of metric names.
            - add_train_metric_weight: List of weights for each metric.
            - more_train_metric: Dictionary of loss functions.

    Returns:
        A dictionary of weighted loss values for each metric.
    """
    all_losses = {}
    for idx, new_metric in enumerate(args.add_train_metric):
        loss_fun_now = args.more_train_metric[new_metric]
        if new_metric == "SSIM":
            # Clip intensity values to ensure data within range
            model_output_here = torch.clamp(model_output, min=args.global_min, max=args.global_max) - args.global_min
            target_here = torch.clamp(target, min=args.global_min, max=args.global_max) - args.global_min
            # Compute SSIM loss
            loss_value_now = loss_fun_now(model_output_here, target_here)
            # Ensure it's not negative
            loss_value_now = torch.clamp(loss_value_now, min=0, max=2)
        else:
            loss_value_now = loss_fun_now(model_output, target)
        all_losses[new_metric] = loss_value_now.mean() * float(args.add_train_metric_weight[idx])
    return all_losses

def random_hidden(t1, t1c, t2, flair, device):
            """
            Randomly zero‑out one modality by setting it to ones,
            then concatenate all four along the channel dimension.
            """
            feats = [t1, t1c, t2, flair]
            # pick random index 0–3
            i = random.randrange(4)
            # Clone the hidden modal
            target = feats[i].clone()
            # replace that tensor with ones
            feats[i] = torch.ones_like(feats[i])
            # concat all four along channel dim
            out = torch.cat(feats, dim=1)
            return out.to(device), target.to(device)

def train(dataLoader_obj, model, optimizer, train_loader_mri, scaler, args):
    """
    Training function.
    Called once per epoch.
    In:
        model: model weights
        optimizer: optimizer for model weight update
        train_loader_mri: dataloader from get_dataloader
    Out:
        Average loss
    """
    #1: set the model to training mode
    model.train()
    A_to_B_losses = {'loss': []} # Initialize the loss dictionary
    total_time = 0

    for idx, new_metric in enumerate(args.add_train_metric):
        A_to_B_losses[new_metric] = []

    # Initialize the tqdm progress bar once
    # Set the number of iteration as the number of mri cases dividing by batch size
    progress_bar = tqdm(train_loader_mri, desc="Training MRI Model")

    crop_foreground = CropForeground() # To get the dimentions for cropping a bouding of of the region which was inpainted
    
    for i, batch in enumerate(progress_bar): 
        t1 = batch["t1"].reshape(-1, 1, *args.patch_size).to(args.device)
        t1c = batch["t1c"].reshape(-1, 1, *args.patch_size).to(args.device)
        t2 = batch["t2"].reshape(-1, 1, *args.patch_size).to(args.device)
        flair = batch["flair"].reshape(-1, 1, *args.patch_size).to(args.device)

        traincondition, target = random_hidden(t1, t1c, t2, flair, args.device)

        # Extract random timestep for training
        t, weights = schedule_sampler.sample(traincondition.shape[0], args.device)

        # Optimize the TDM network
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            # Compute the losses
            all_loss, target, model_output = train_diffusion.training_losses(A_to_B_model, target, traincondition, t, train_metric=args.train_metric)
            A_to_B_loss = (all_loss["loss"] * weights).mean() # Loss from the train_metric (default is MSE loss)

            # Compute MAE and MSE for the region inpainted only!
            # The SSIM uses the entire patch
            # Add the weighted losses from additional metrics
            # If none are provided, this will be skipped
            addicional_losses = get_additional_losses(
                target=target, 
                model_output=model_output,
                args=args)

            for add_loss in addicional_losses:
                A_to_B_loss += addicional_losses[add_loss]
                A_to_B_losses[add_loss].append(addicional_losses[add_loss].mean().detach().cpu().numpy())
                wandb.log({add_loss: addicional_losses[add_loss].item()}) # wandb save

            wandb.log({"DDPM loss": (all_loss["loss"] * weights).mean().item()}) # wandb save
            wandb.log({"Complete Loss": A_to_B_loss.item()}) # wandb save

            # Check for NaN values in all_loss
            if torch.isnan(A_to_B_loss): 
                print("NaN detected in loss, skipping this batch.")
                del all_loss
                del A_to_B_loss
                continue

            # Append the loss value for tracking
            A_to_B_losses["loss"].append(all_loss["loss"].mean().detach().cpu().numpy())

        # Update tqdm with loss info
        progress_bar.set_postfix(base_loss=all_loss["loss"].mean().detach().cpu().numpy())

        for name, param in model.named_parameters(): 
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradients for {name}")

        # Backpropagation step with automatic mixed precision
        # A_to_B_loss = MSE_patch + SSIM_patch + MSE_roi + MAE_roi
        scaler.scale(A_to_B_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    if np.isnan(A_to_B_losses["loss"]).any(): 
        print("NaN values detected in loss history. Check earlier stages.")

    #6: print out total time 
    print("Total time per sample is: "+str(time.time()-total_time))

    return A_to_B_losses, model, optimizer, scaler

def diffusion_sampling(condition, model):
    sampled_images = val_diffusion.ddim_sample_loop_inpaint(
        model,
        (condition.shape[0], 1, condition.shape[2], condition.shape[3],condition.shape[4]),
        condition=condition,
        clip_denoised=args.clip_denoised
        )
    return sampled_images

def pad_if_smaller_symmetric(in_tensor, patch_size):
    """
    Pads a 5D tensor symmetrically along its depth, height, and width dimensions 
    if any of these dimensions are smaller than the corresponding patch size.

    Parameters:
        in_tensor (torch.Tensor): The input tensor with shape (B, C, D, H, W), 
                                  where B is the batch size, C is the number of channels, 
                                  D is the depth, H is the height, and W is the width.
        patch_size (tuple of int): A tuple (D_patch, H_patch, W_patch) specifying the 
                                   minimum required size for the depth, height, and width 
                                   dimensions of the tensor.

    Returns:
        tuple:
            - torch.Tensor: The padded tensor with the same batch size and channels, 
                            but with depth, height, and width dimensions padded symmetrically 
                            to meet or exceed the specified patch size.
            - tuple of int: A tuple (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2) 
                            representing the amount of padding applied to each side 
                            of the width, height, and depth dimensions, respectively.

    """
    pad_d = max(patch_size[0] - in_tensor.shape[2], 0)
    pad_h = max(patch_size[1] - in_tensor.shape[3], 0)
    pad_w = max(patch_size[2] - in_tensor.shape[4], 0)

    pad_d1, pad_d2 = pad_d // 2, pad_d - pad_d // 2
    pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
    pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2

    # F.pad expects (W_before, W_after, H_before, H_after, D_before, D_after)
    padding = (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2)

    return F.pad(in_tensor, padding), padding

def unpad_if_smaller_symmetric(in_tensor, padding):
    """
    Removes symmetric padding from a tensor if padding was applied.

    This function removes padding from the input tensor along the depth, height, 
    and width dimensions based on the specified padding values. The padding is 
    assumed to be symmetric, and the function ensures that the tensor is sliced 
    correctly even if some padding values are zero.

    Args:
        in_tensor (torch.Tensor): The input tensor to unpad. Expected to have 
            dimensions (batch_size, channels, depth, height, width).
        padding (tuple): A tuple of six integers specifying the padding values 
            in the order (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2), where:
            - pad_w1, pad_w2: Padding on the width dimension (start, end).
            - pad_h1, pad_h2: Padding on the height dimension (start, end).
            - pad_d1, pad_d2: Padding on the depth dimension (start, end).

    Returns:
        torch.Tensor: The unpadded tensor with the same number of dimensions as 
        the input tensor but with reduced size along the depth, height, and 
        width dimensions if padding was applied.
    """
    pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2 = padding

    if pad_d1 + pad_d2 > 0:
        in_tensor = in_tensor[:, :, pad_d1:-pad_d2 if pad_d2 > 0 else None, :, :]
    if pad_h1 + pad_h2 > 0:
        in_tensor = in_tensor[:, :, :, pad_h1:-pad_h2 if pad_h2 > 0 else None, :]
    if pad_w1 + pad_w2 > 0:
        in_tensor = in_tensor[:, :, :, :, pad_w1:-pad_w2 if pad_w2 > 0 else None]
    return in_tensor

def evaluate(model, epoch, val_loader_mri, inferer, args):
    """
    Run the evaluate function will translate the MRI to CT.
    The result will be saved in nii.gz format
    """
    model.eval()
    loss_all = []
    crop_foreground = CropForeground() # To get the dimentions for cropping a bouding of of the region which was inpainted
    
    if (epoch+1) % args.val_interval == 0: 
        with torch.no_grad():
            for i, batch in enumerate(tqdm(islice(val_loader_mri, 1), desc="1 Random cases on validation:", total=1)):
                t1 = batch["t1_fullres"].to(args.device)
                t1c = batch["t1c_fullres"].to(args.device)
                t2 = batch["t2_fullres"].to(args.device)
                flair = batch["flair_fullres"].to(args.device)

                condition, target = random_hidden(t1, t1c, t2, flair, args.device)
                
                # Voided MRI
                mri_image_batch = target[0][0].cpu().numpy()  
                # Rescale the values from z-score to original
                nib.save(nib.Nifti1Image(
                        mri_image_batch,
                        affine=np.eye(4)),
                        f'{args.path_checkpoint}/scans/{epoch}_{i}_target.nii.gz'
                    )
            
                # Prediction
                # condition is the input voided MRI
                # target is the MRI scan          
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
                
                nib.save(
                    nib.Nifti1Image(predict_ct_image_batch, affine=np.eye(4)), 
                    f'{args.path_checkpoint}/scans/{epoch}_{i}_validation.nii.gz'
                    )
            
                if i >= 0: # Uses only 3 cases for validation
                    break
    with torch.no_grad():
        pbar = tqdm(val_loader_mri, desc="Validation")
        for i, batch in enumerate(pbar):
            t1 = batch["t1"].reshape(-1, 1, *args.patch_size).to(args.device)
            t1c = batch["t1c"].reshape(-1, 1, *args.patch_size).to(args.device)
            t2 = batch["t2"].reshape(-1, 1, *args.patch_size).to(args.device)
            flair = batch["flair"].reshape(-1, 1, *args.patch_size).to(args.device)

            condition, target = random_hidden(t1, t1c, t2, flair, args.device)
            
            # Predict patch
            #t, weights = schedule_sampler.sample(condition.shape[0], device)
            
            # Select timesteps for better validation
            # Fixed tensors
            t1 = torch.randint(1, 2, (condition.shape[0],), device=args.device)   # 1 

            # Random tensors within specified ranges
            t2 = torch.randint(2, 6, (condition.shape[0],), device=args.device)   # 2 to 5
            t3 = torch.randint(5, 11, (condition.shape[0],), device=args.device)  # 5 to 10
            t4 = torch.randint(10, 16, (condition.shape[0],), device=args.device) # 10 to 15
            t5 = torch.randint(15, 21, (condition.shape[0],), device=args.device) # 15 to 20
            t6 = torch.randint(20, 25, (condition.shape[0],), device=args.device) # 20 to 24
            t_all = [t1,t2,t3,t4,t5,t6]
            for t in t_all:
                _, _, sampled_images = train_diffusion.training_losses(A_to_B_model, target, condition, t, train_metric=args.train_metric)
                # MAE
                MAE_value = eval_MAE(sampled_images, target)
                # MSE
                MSE_value = eval_MSE(sampled_images, target)
                #
                ## Computed on the entire patch
                # Norm to compute PSNR and SSIM
                sampled_images_here = sampled_images.clone()
                target_here = target.clone() 
                # Clip between min and max
                sampled_images_here = sampled_images_here.clamp(args.global_min, args.global_max)
                target_here = target_here.clamp(args.global_min, args.global_max)
                # Making sure it is within the range and that the range is correct
                sampled_images_here = sampled_images_here-args.global_min
                target_here = target_here-args.global_min
                PSNR_value = eval_PSNR(sampled_images_here, target_here)
                SSIM_value = args.more_train_metric['SSIM'](sampled_images_here, target_here) 

                pbar.set_postfix({
                    "t_0": t[0].item(),
                    "mae_loss": f"{MAE_value.mean().cpu().numpy():.4f}"
                })

                if torch.isnan(MAE_value).any(): 
                    print("NaN detected in validation loss. MAE_value.")
                if torch.isnan(MSE_value).any(): 
                    print("NaN detected in validation loss. MSE_value.")
                if torch.isnan(PSNR_value).any(): 
                    print("NaN detected in validation loss. PSNR_value.")
                if torch.isnan(SSIM_value).any(): 
                    print("NaN detected in validation loss. SSIM_value.")
                
                loss_all.append({
                    "MAE": MAE_value.cpu().numpy(),
                    "MSE": MSE_value.cpu().numpy(),
                    "PSNR": PSNR_value.cpu().numpy(),
                    "SSIM": SSIM_value.cpu().numpy(),
                })
        
            
        return loss_all

def warm_up_model(A_to_B_model, freeze_epochs, args):
        """
        Perform a warm-up phase for the given model by training it for a specified number of epochs 
        with frozen pre-trained parameters. The function logs the training loss for each warm-up epoch and saves 
        a checkpoint of the warmed-up model.

        Args:
            A_to_B_model (torch.nn.Module): The model to be warmed up.
            

        Returns:
            tuple: A tuple containing the warmed-up model (`torch.nn.Module`).

        Notes:
            - The number of warm-up epochs is determined by the `args.freeze_epochs` parameter.
            - Training is performed using the `train` function, which should be defined elsewhere.
            - The training loss for each warm-up epoch is logged using `wandb`.
            - A checkpoint of the warmed-up model is saved to the path specified by `args.path_checkpoint`.

        Checkpoint Contents:
            - `model_state_dict`: The state dictionary of the warmed-up model.
            - `optimizer_state_dict`: The state dictionary of the optimizer.
            - `epoch`: The current epoch number.
            - `best_loss`: 0

        Dependencies:
            - `np.nanmean`: Used to compute the average loss while ignoring NaN values.
            - `wandb.log`: Used for logging training metrics.
            - `torch.save`: Used for saving the model checkpoint.
            - `args`: A global variable containing configuration parameters such as `freeze_epochs` 
              and `path_checkpoint`.
            - `train_dataloader`: A global variable representing the training data loader.
            - `train`: A function that performs one epoch of training and returns the losses, updated 
              model, and optimizer.
        """
        #optimizer = torch.optim.AdamW(A_to_B_model.parameters(), lr=2e-5, weight_decay = 1e-4)
        optimizer = torch.optim.AdamW(
            A_to_B_model.parameters(),  # frozen encoder params will be skipped
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

        scaler = torch.amp.GradScaler("cuda")

        for warm_up_epoch in range (freeze_epochs):
            print('Warmup epoch:', warm_up_epoch)
            # Build the training function. Run the training function once = one epoch
            A_to_B_losses, A_to_B_model, optimizer, scaler = train(
                model=A_to_B_model, 
                optimizer=optimizer, 
                train_dataloader=train_dataloader,
                scaler=scaler,
                )
            average_loss_train = np.nanmean(A_to_B_losses)
            wandb.log({"Loss warmup_epoch": average_loss_train.item(), "warmup_epoch": warm_up_epoch}) # wandb save
            print('Averaged warmup loss is: '+ str(average_loss_train))
        checkpoint = {
                    'model_state_dict': A_to_B_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': warm_up_epoch,
                    'best_loss': 0
                }
        torch.save(checkpoint, join(args.path_checkpoint, "model", 'A_to_B_model_warmed_up.pt'))
        return A_to_B_model

def warmup_step(A_to_B_model, filtered_dict, not_matching_keys, args):
    """
    Perform a warm-up phase for the given model by freezing and unfreezing specific layers 
    in a stepwise manner. This function ensures that pre-trained layers are utilized effectively 
    while allowing new layers to adapt to the task.

    Args:
        A_to_B_model (torch.nn.Module): The model to be warmed up.
        filtered_dict (dict): Dictionary of pre-trained weights that match the model's layers.
        not_matching_keys (list): List of keys that did not match during weight loading.

    Returns:
        torch.nn.Module: The warmed-up model with updated weights.

    Notes:
        - The warm-up process is divided into two phases:
            1. Freeze pre-trained layers and train only new layers.
            2. Unfreeze decoder layers and train them along with the new layers.
        - The number of epochs for each phase is determined by `args.freeze_epochs`.
        - The `freeze_layers` function is used to freeze pre-trained layers.
        - The `warm_up_model` function is used to train the model during each phase.
    """
    # Phase 1: Freeze pre-trained layers and train only new layers
    for name, param in A_to_B_model.named_parameters():
        param.requires_grad = True  # Ensure all layers are initially learnable
    A_to_B_model = freeze_layers(A_to_B_model, filtered_dict, not_matching_keys)
    A_to_B_model = warm_up_model(A_to_B_model, freeze_epochs=args.freeze_epochs // 2)

    # Phase 2: Unfreeze decoder layers and train them along with new layers
    for name, param in A_to_B_model.named_parameters():
        if "decoder" in name:  # Unfreeze decoder layers
            param.requires_grad = True
    A_to_B_model = warm_up_model(A_to_B_model, freeze_epochs=args.freeze_epochs)

    # Phase 3: Unfreeze all layers
    for name, param in A_to_B_model.named_parameters():
        param.requires_grad = True  # Ensure all layers are initially learnable

    return A_to_B_model


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
    parser.add_argument('--path_checkpoint', type=str, default='../../results/', help='Path to save model checkpoints')
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
    parser.add_argument('--use_global_stats', action='store_true', help='Linearly normalize the intensities to -1 and 1 (from the z-score).')
    parser.add_argument('--noise_schedule', type=str, default="linear", help='Noise scheduler. linear (default) or cosine')
    parser.add_argument('--use_cosine_scheduler', action='store_true', help='If use learning rate decay to 1e-6.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    seed_value = 42
    set_complete_seed(seed_value)
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    name = f"MC-IDDPM_{args.batch_size_train}_{args.n_epochs}_timestep_{args.timestep_respacing}_patchsize_{args.patch_num}_{args.network}_{args.train_metric}"
    for add_train in args.add_train_metric:
        name += "_"+str(add_train)
 
    for patch_size_idx in args.patch_size:
        name += "_"+str(patch_size_idx)

    if args.use_global_stats:
        name += f"_global_stats"

    if args.prob != 0:
        name += f"_DA_{args.prob}"
    args.path_checkpoint = join(args.path_checkpoint, name)
    wandb.init(project="BraTS2025_Syn", name=name, dir=args.path_checkpoint)
    args.path_checkpoint = wandb.run.dir

    # Create checkpoint folder
    os.makedirs(f"{args.path_checkpoint}/scans",exist_ok=True)
    os.makedirs(f"{args.path_checkpoint}/model",exist_ok=True)

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
        print(f"Using global stats for z-score: {args.use_global_stats}")
        print(f"Using device: {args.device}")
        print("####################################")

    ########################################
    ######### Define data loader ##########
    ########################################
    dataLoader_obj = MonaiDataLoader(
        spatial_size=args.patch_size,
        number_of_patches=args.patch_num,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
        prob=args.prob,
        use_global_stats=args.use_global_stats,
        json_file='/homes/andre.ferreira/BraTS2025/Dataset/Global-Synthesis_Train/data_split.json')

    
    train_loader_mri  = dataLoader_obj.get_loaders_train()
    
    val_loader_mri  = dataLoader_obj.get_loaders_val()

    ##################################################################
    ############ Loss functions and validation metrics ###############
    ##################################################################
   
    # For the evaluation step (NormalizeIntensityd has a variable range)
    args.clip_denoised = False
    
    # Used for training
    args.more_train_metric = {}
    add_train_metric = list(args.add_train_metric)

    args.more_train_metric['MSE'] =  torch.nn.MSELoss(reduction='none') 

    for add_metric in args.add_train_metric:
        ### Adding the MAE loss function
        if add_metric=='MAE':
            print(f"Using MAE as loss function")
            args.more_train_metric['MAE'] =  torch.nn.L1Loss(reduction='none')
            add_train_metric.remove('MAE')
        ### Adding the SSIM loss function
        if add_metric=='SSIM':
            print(f"Using SSIM as loss function")
            args.global_min = -1 # 6 STDs
            args.global_max = 1
            data_range = args.global_max - (args.global_min)
            print(f"Set data range to: {data_range}")
            args.more_train_metric['SSIM'] = SSIMLoss(
                                                spatial_dims=3,
                                                data_range=data_range,
                                                kernel_type='gaussian',
                                                win_size=11,
                                                kernel_sigma=1.5,
                                                k1=0.01,
                                                k2=0.03,
                                                reduction='mean')
            add_train_metric.remove('SSIM')

    print(f"Metrics used for training: {args.train_metric} and {args.more_train_metric}. Not used: {add_train_metric}")

     # Used for evaluation 
    
    eval_MAE = torch.nn.L1Loss()
    eval_MSE = torch.nn.MSELoss()
    eval_PSNR = PSNRMetric(
        max_val=data_range, 
        reduction="mean", 
        get_not_nans=False)

    ########################################
    ###### Build the MC-IDDPM process ######
    ########################################
    # The MC-IDDPM process is a combination of the DDPM and the transformer network. 
    # The DDPM is used to denoise the MRI image
    train_diffusion, val_diffusion, schedule_sampler = get_diffusion(
        timestep_respacing=args.timestep_respacing,
        timestep_respacing_val=args.timestep_respacing_val,
        args=args
        )
    A_to_B_model, filtered_dict, not_matching_keys = get_model(
        device=args.device,
        args=args
        )
    
    # Print the number of parameters in the model
    if args.verbose:
        print('parameter number is '+str(sum(p.numel() for p in A_to_B_model.parameters())))
    
    # optimizer = torch.optim.AdamW(A_to_B_model.parameters(), lr=2e-5, weight_decay=1e-4) # Old!
    optimizer = torch.optim.AdamW(
            A_to_B_model.parameters(),  # frozen encoder params will be skipped
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

    if args.use_cosine_scheduler:
        cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.n_epochs,
                eta_min=1e-6
            )

    scaler = torch.amp.GradScaler("cuda")
    
    ########################################
    ############## Inference ###############
    ########################################
    # Use the sliding window  method to translate the whole MRI to CT volume. Must used it.
    # For example, if your whole volume is 64x64x64, and our window size is 64x64x4, so the function will automatically sliding down
    # the whole volume with a certain overlapping ratio
    # The window size (args.patch_size) is shown in the "Build the data loader using the monai library" section.
    # args.patch_size: the size of sliding window
    # img_num: the number of sliding window in each process, only related to your gpu memory, it will still run through the whole volume
    # overlap: the overlapping ratio
    inferer = SlidingWindowInferer(
        roi_size=(args.patch_size[0], args.patch_size[1], args.patch_size[2]), 
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap, 
        mode=args.overlap_mode, 
        progress=True
        )
    smoother = EMASmoother(alpha=0.90)

    ########################################
    ############ Training Loop #############
    ########################################
    if args.resume!=None:
        checkpoint = torch.load(join(args.resume), weights_only=False)
        A_to_B_model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Retrieve the epoch and best loss
        #begin_epoch = checkpoint['epoch'] #TODO
        #best_loss = checkpoint['best_loss'] #TODO
        #best_psnr = checkpoint['best_psnr']  #TODO
        best_loss = 10000 #TODO
        begin_epoch = 0 #TODO
        best_psnr=0 # TODO
        print(f"Loaded from: {args.resume}")
    else:
        best_loss = 10000
        best_psnr = 0
        begin_epoch = 0
        print("Start training from scratch")

    pacience_counter = 0

    for epoch in range(begin_epoch, args.n_epochs): 
        print('Epoch:', epoch)
        start_time = time.time() 

        # Build the training function. Run the training function once = one epoch
        A_to_B_losses, A_to_B_model, optimizer, scaler = train(
            dataLoader_obj=dataLoader_obj,
            model=A_to_B_model, 
            optimizer=optimizer, 
            train_loader_mri=train_loader_mri,
            scaler=scaler,
            args=args
            )
        if args.use_cosine_scheduler:
            cosine_scheduler.step()
        wandb.log({
            "epoch": epoch, # Log the actual epoch number to the 'epoch' metric
            "lr": optimizer.param_groups[0]['lr'] # Use optimizer here, not opt_ae if it's passed as arg
            })

        average_loss_train = 0
        for idx, new_metric in enumerate(args.add_train_metric):
            wandb.log({f"Epoch_{new_metric}": np.nanmean(A_to_B_losses[new_metric]), "epoch": epoch})
            average_loss_train += np.nanmean(A_to_B_losses[new_metric])
            

        wandb.log({f"Epoch_{args.train_metric}": np.nanmean(A_to_B_losses['loss']), "epoch": epoch})
        average_loss_train += np.nanmean(A_to_B_losses['loss'])

        wandb.log({"Loss epoch": average_loss_train.item(), "epoch": epoch}) # wandb save
        print('Averaged loss is: '+ str(average_loss_train))
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        
        # Validation every epoch
        print("Evaluating...")
        loss_val = evaluate(
            model=A_to_B_model, 
            epoch=epoch, 
            val_loader_mri=val_loader_mri,
            inferer=inferer,
            args=args
            )
        try:
            maes = [d['MAE'].mean().item() for d in loss_val]
            mean_mae = np.mean(maes)
        except:
            print(f"Error Computing the mean MAE for validation")
            mean_mae = 1

        try:
            mses = [d['MSE'].mean().item() for d in loss_val]
            mean_mse = np.mean(mses)
        except:
            print(f"Error Computing the mean MSE for validation")
            mean_mse = 1

        try:
            psnrs = [d['PSNR'].mean().item() for d in loss_val]
            mean_psnr = np.mean(psnrs)
        except:
            print(f"Error Computing the mean PSNR for validation")
            mean_psnr = 0

        try:
            ssims = [d['SSIM'].mean().item() for d in loss_val]
            mean_ssim = np.mean(ssims)
        except:
            print(f"Error Computing the mean SSIM for validation")
            mean_ssim = 0

        wandb.log({f"Val MAE": mean_mae, "epoch": epoch}) # wandb save
        wandb.log({f"Val MSE": mean_mse, "epoch": epoch}) # wandb save
        wandb.log({f"Val PSNR": mean_psnr, "epoch": epoch}) # wandb save
        wandb.log({f"Val SSIM": mean_ssim, "epoch": epoch}) # wandb save

        smoothed_val_losses = smoother.update(mean_mae)

        if smoothed_val_losses < best_loss:
            print(f'New validation best 🎉 Save the latest best model. smoothed_val_mse={smoothed_val_losses} | Real mean_mae={mean_mae}')
            pacience_counter = 0
            checkpoint = {
                'model_state_dict': A_to_B_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'best_psnr': mean_psnr
            }
            # Save the checkpoint dictionary
            torch.save(checkpoint, join(args.path_checkpoint, "model", 'A_to_B_model_best.pt'))
            best_loss = smoothed_val_losses
        else:
            print(f"Epoch smooth validation loss: {smoothed_val_losses} | Epoch validation mean_mae: {mean_mae} | Best loss: {best_loss}")
            if epoch>(args.n_epochs//3)*2: # Only start counting pacience after 2/3 of the epochs ran
                pacience_counter += 1
                print(f"Pacience increased by 1. Value {pacience_counter} out of {args.pacience} epochs. Epoch smooth validation loss: {smoothed_val_losses} | Epoch validation mean_mae: {mean_mae} | Best loss: {best_loss}")

        # Save model with best PSNR
        if mean_psnr > best_psnr:
            print('Save the latest best model psnr')
            checkpoint = {
                'model_state_dict': A_to_B_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'best_psnr': mean_psnr
            }
            # Save the checkpoint dictionary
            torch.save(checkpoint, join(args.path_checkpoint, "model", 'A_to_B_model_best_psnr.pt'))
            best_psnr = mean_psnr
        
        # Save every 10 epochs 
        if (epoch+1) % 10 == 0: 
            checkpoint = {
                    'model_state_dict': A_to_B_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'best_psnr': mean_psnr
                }
            torch.save(checkpoint, join(args.path_checkpoint, "model", 'A_to_B_model_latest.pt')) 
        
        if pacience_counter >= args.pacience:
            print(f"The validation results are not improving for {args.pacience} epochs!")
            print(f"Stopping training")
            break
    wandb.finish()
    