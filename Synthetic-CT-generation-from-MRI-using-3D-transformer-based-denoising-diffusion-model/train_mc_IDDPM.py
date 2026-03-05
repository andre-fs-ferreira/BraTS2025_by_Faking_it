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
            in_channels=3, # noised, voided_mri_batch, corrected_mask_healthy_batch
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

def get_additional_losses(target, model_output, mask, start, end, args):
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
            # Crop to focus more on the region that was inpainted
            # Removed to avoid instability.
            #model_output_here = model_output_here[:,:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            #target_here = target_here[:,:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            # Compute SSIM loss
            loss_value_now = loss_fun_now(model_output_here, target_here)
            # Ensure it's not negative
            loss_value_now = torch.clamp(loss_value_now, min=0, max=2)
        else:
            # Mask wise loss
            masked_squared_diff = loss_fun_now(model_output, target)
            masked_squared_diff = masked_squared_diff * mask.float()
            sum_masked_squared_diff = torch.sum(masked_squared_diff)
            num_active_elements = torch.sum(mask.float())
            loss_value_now = sum_masked_squared_diff / num_active_elements
        
        if torch.isnan(loss_value_now).any():
            print(f"Nan values computing {new_metric}. Setting value to 1.")
            loss_value_now = torch.tensor(1.0, device=loss_value_now.device)
        if torch.isinf(loss_value_now).any():
            print(f"Inf values computing {new_metric}. Setting value to 1.")
            loss_value_now = torch.tensor(1.0, device=loss_value_now.device)
        all_losses[new_metric] = loss_value_now.mean() * float(args.add_train_metric_weight[idx])
    return all_losses

def random_foreground_crop_batch(
    ct_fullres: torch.Tensor,      # (B, C, H, W, D)
    mri_fullres: torch.Tensor,     # (B, C, H, W, D)
    mask: torch.Tensor,            # (B, C, H, W, D)
    crop_size=(128, 128, 32)
    ):
    """
    Perform random foreground cropping over a batch.
    Returns cropped volumes and coordinates for each item in batch.
    """
    B, C, H, W, D = ct_fullres.shape
    crop_H, crop_W, crop_D = crop_size

    cropped_ct = torch.empty((B, C, crop_H, crop_W, crop_D), dtype=ct_fullres.dtype, device=ct_fullres.device)
    cropped_mri = torch.empty_like(cropped_ct)
    crop_coords = torch.empty((B, 6), dtype=torch.int32, device=ct_fullres.device)

    mask_squeezed = mask[:, 0]  # Assuming C=1 for the mask.

    for b in range(B):
        fg_indices = (mask_squeezed[b] == 1).nonzero(as_tuple=False)
        if fg_indices.size(0) == 0:
            raise ValueError(f"Sample {b} contains no foreground voxels.")

        idx = torch.randint(0, fg_indices.size(0), (1,), device=ct_fullres.device)
        center_y, center_x, center_z = fg_indices[idx].squeeze(0)

        start_y = torch.clamp(center_y - crop_H // 2, min=0, max=H - crop_H)
        start_x = torch.clamp(center_x - crop_W // 2, min=0, max=W - crop_W)
        start_z = torch.clamp(center_z - crop_D // 2, min=0, max=D - crop_D)

        end_y = start_y + crop_H
        end_x = start_x + crop_W
        end_z = start_z + crop_D

        cropped_ct[b] = ct_fullres[b, :, start_y:end_y, start_x:end_x, start_z:end_z]
        cropped_mri[b] = mri_fullres[b, :, start_y:end_y, start_x:end_x, start_z:end_z]
        crop_coords[b] = torch.tensor([start_y, start_x, start_z, end_y, end_x, end_z], dtype=torch.int32, device=ct_fullres.device)

    return cropped_ct, cropped_mri, crop_coords

def check_and_clamp_bounding_box_shape(start_coords, end_coords, image_dims, min_dim=12):
    """
    Adjusts and clamps bounding box coordinates to ensure minimum dimensions
    and stay within image boundaries.

    Args:
        start_coords (list or np.array): [min_z, min_y, min_x] lowest coordinates of the bbox.
        end_coords (list or np.array): [max_z, max_y, max_x] highest coordinates of the bbox.
        image_dims (tuple or list): (D, H, W) or (Z, Y, X) dimensions of the original image/patch.
        min_dim (int): The minimum desired dimension for each axis.

    Returns:
        tuple: (adjusted_start, adjusted_end)
    """
    adjusted_start = np.array(start_coords, dtype=int)
    adjusted_end = np.array(end_coords, dtype=int)

    for i in range(3): # Iterate through Z, Y, X dimensions
        current_dimension = adjusted_end[i] - adjusted_start[i]

        if current_dimension < min_dim:
            # Calculate how much to add
            add_needed = min_dim - current_dimension

            # Try to expand equally on both sides
            temp_start = adjusted_start[i] - add_needed // 2
            temp_end = adjusted_end[i] + (add_needed - add_needed // 2)

            # Clamp to image boundaries
            # Handle the start (min_coord)
            if temp_start < 0:
                adjusted_start[i] = 0
                adjusted_end[i] = min(image_dims[i], adjusted_end[i] + (0 - temp_start)) # Shift end if start was clamped
            else:
                adjusted_start[i] = temp_start

            # Handle the end (max_coord)
            if temp_end > image_dims[i]:
                adjusted_end[i] = image_dims[i]
                adjusted_start[i] = max(0, adjusted_start[i] - (temp_end - image_dims[i])) # Shift start if end was clamped
            else:
                adjusted_end[i] = temp_end

            # Final check: After clamping, is the dimension still at least min_dim?
            # This handles cases where the original bbox was too close to the edge.
            final_dim_after_clamp = adjusted_end[i] - adjusted_start[i]
            if final_dim_after_clamp < min_dim:
                # If still too small, it means the original bbox was too close to an edge
                # and couldn't expand symmetrically. We must force the size by expanding
                # towards the available space.
                if adjusted_start[i] == 0: # Clamped at start, try to expand end
                    adjusted_end[i] = min(image_dims[i], adjusted_start[i] + min_dim)
                elif adjusted_end[i] == image_dims[i]: # Clamped at end, try to expand start
                    adjusted_start[i] = max(0, adjusted_end[i] - min_dim)
                else: # Should not happen if previous logic is correct unless min_dim > image_dims[i]
                    # This case means the entire image dimension is less than min_dim
                    # In this scenario, we just take the full image dimension.
                    adjusted_start[i] = 0
                    adjusted_end[i] = image_dims[i]

    return adjusted_start.tolist(), adjusted_end.tolist()

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
        voided_mri = batch["voided_mri"].reshape(-1, 1, *args.patch_size).to(args.device)
        healthy_region_mask_corrected = batch["healthy_region_mask_corrected"].reshape(-1, 1, *args.patch_size).to(args.device)
        target = batch["target"].reshape(-1, 1, *args.patch_size).to(args.device) # Regions healthy

        if torch.sum(healthy_region_mask_corrected)==0:
            print(f"Skipping batch {i} due to no healthy region to inpaint.")
            continue # Skip if there is no healthy region to inpaint
        
        if torch.isnan(voided_mri).any() or torch.isinf(voided_mri).any() or \
            torch.isnan(healthy_region_mask_corrected).any() or torch.isinf(healthy_region_mask_corrected).any() or \
            torch.isnan(target).any() or torch.isinf(target).any():
                print(f"NaN or Inf detected in input data at batch {i}")
                continue

        traincondition = torch.cat(
            (voided_mri, healthy_region_mask_corrected),
            dim=1
        ).to(args.device)

        # Start and end of the bounding box # Removed
        #start, end = crop_foreground.compute_bounding_box(img=healthy_region_mask_corrected[0].cpu().numpy())
        #start, end = check_and_clamp_bounding_box_shape(start, end, args.patch_size, min_dim=12)
        start, end = 0, 0
        # Extract random timestep for training
        t, weights = schedule_sampler.sample(traincondition.shape[0], args.device)

        # Optimize the TDM network
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            flag_ok = True
            try:
                with torch.autograd.detect_anomaly():
                    all_loss, target, model_output = train_diffusion.training_losses(A_to_B_model, target, traincondition, t, train_metric=args.train_metric)
                    A_to_B_loss = (all_loss["loss"] * weights).mean() # Loss from the train_metric (default is MSE loss)

                    if torch.isnan(model_output).any() or torch.isinf(model_output).any():
                        print(f"NaN or inf detected in model_output at batch {i}, skipping this batch.")
                        flag_ok = False

                    if torch.isnan(A_to_B_loss).any() or torch.isinf(A_to_B_loss).any():
                        print(f"NaN or inf detected in A_to_B_loss at batch {i}, skipping this batch. 1.")
                        flag_ok = False
                    
                    if torch.isnan(all_loss["mse"]).any() or torch.isinf(all_loss["mse"]).any():
                        print(f"NaN or inf detected in all_loss[mse] at batch {i}, skipping this batch. 1.")
                        flag_ok = False
                    
                    if torch.isnan(all_loss["vb"]).any() or torch.isinf(all_loss["vb"]).any():
                        print(f"NaN or inf detected in all_loss[vb] at batch {i}, skipping this batch. 1.")
                        flag_ok = False

                    # Compute MAE and MSE for the region inpainted only!
                    # The SSIM uses the entire patch
                    # Add the weighted losses from additional metrics
                    # If none are provided, this will be skipped
                    addicional_losses = get_additional_losses(target, model_output, mask=healthy_region_mask_corrected, start=start, end=end, args=args)
                    for add_loss in addicional_losses:
                        A_to_B_loss += addicional_losses[add_loss]
                        if torch.isnan(A_to_B_loss) or torch.isinf(A_to_B_loss).any(): 
                            print(f"NaN or inf detected in loss, skipping this batch. 2. {add_loss}")
                            flag_ok = False
                        A_to_B_losses[add_loss].append(addicional_losses[add_loss].mean().detach().cpu().numpy())
                        wandb.log({add_loss: addicional_losses[add_loss].item()}) # wandb save

                    wandb.log({"DDPM loss": (all_loss["loss"] * weights).mean().item()}) # wandb save
                    wandb.log({"MSE loss": (all_loss["mse"]).mean().item()}) # wandb save
                    wandb.log({"vb loss": (all_loss["vb"]).mean().item()}) # wandb save
                    wandb.log({"Complete Loss": A_to_B_loss.item()}) # wandb save

                    # Check for NaN values in all_loss
                    if torch.isnan(A_to_B_loss): 
                        print("NaN detected in loss, skipping this batch. 3. ")
                        flag_ok = False

                    # Append the loss value for tracking
                    A_to_B_losses["loss"].append(all_loss["loss"].mean().detach().cpu().numpy())
                    if flag_ok==False:
                        print(f"Batch {i}: Input stats: voided_mri min={voided_mri.min().item()}, max={voided_mri.max().item()}, mean={voided_mri.mean().item()}, std={voided_mri.std().item()}")
                        print(f"Batch {i}: Target stats: min={target.min().item()}, max={target.max().item()}, mean={target.mean().item()}, std={target.std().item()}")
                        print(f"Batch {i}: Mask stats: min={healthy_region_mask_corrected.min().item()}, max={healthy_region_mask_corrected.max().item()}")
                        print(f"Batch {i}: Timesteps = {t}")
                        torch.save({"voided_mri": voided_mri, "target": target, "mask": healthy_region_mask_corrected, "timesteps": t}, f"problematic_batch_{i}.pt")
                        
                        

            except RuntimeError as e:
                print(f"Batch {i}: Anomaly detected: {e}")
                print(f"Batch {i}: Input stats: voided_mri min={voided_mri.min().item()}, max={voided_mri.max().item()}, mean={voided_mri.mean().item()}, std={voided_mri.std().item()}")
                print(f"Batch {i}: Target stats: min={target.min().item()}, max={target.max().item()}, mean={target.mean().item()}, std={target.std().item()}")
                print(f"Batch {i}: Mask stats: min={healthy_region_mask_corrected.min().item()}, max={healthy_region_mask_corrected.max().item()}")
                print(f"Batch {i}: Timesteps = {t}")
                torch.save({"voided_mri": voided_mri, "target": target, "mask": healthy_region_mask_corrected, "timesteps": t}, f"problematic_batch_{i}.pt")
                del all_loss, target, model_output
                torch.cuda.empty_cache()
                continue
            

        # Update tqdm with loss info
        progress_bar.set_postfix(
            base_loss=all_loss["loss"].mean().detach().cpu().numpy(),
            model_output_max=model_output.max().item(),
            model_output_min=model_output.min().item()
        )

        # Backpropagation step with automatic mixed precision
        # A_to_B_loss = MSE_patch + SSIM_patch + MAE_mask
        # 1) Scale and backward as before
        print(f"Batch {i}: Scaled loss = {scaler.get_scale() * A_to_B_loss.item():.6f}")
        scaler.scale(A_to_B_loss).backward()

        # 2) Unscale the gradients back to FP32
        scaler.unscale_(optimizer)

        # 3) Measure norm before clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        print(f"Batch {i}: Gradient norm before clipping = {grad_norm_before:.6f}")

        # 4) Clip to your chosen max_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        print(f"Batch {i}: Gradient norm after clipping = {grad_norm_after:.6f}")

        # 5) Guard: check all grads are finite
        all_finite = True
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f"Batch {i}: NON‑FINITE gradient in {name}, skipping optimizer.step()")
                all_finite = False
                break

        # 6) Guard: check weights are still finite
        if all_finite:
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all():
                    print(f"Batch {i}: NON‑FINITE weight in {name}, skipping optimizer.step()")
                    all_finite = False
                    break

        # 7) Only step & update scaler if everything is finite
        if all_finite:
            scaler.step(optimizer)
        else:
            # reset any corrupted state to avoid poisoning future steps
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state.get(p, {})
                    for k, v in state.items():
                        if torch.is_tensor(v) and not torch.isfinite(v).all():
                            optimizer.state[p][k] = torch.zeros_like(v)
            print(f"Batch {i}: optimizer.step() SKIPPED and state reset. Timesteps with error used: {t}")

        # 8) Finally update the scaler (always)
        scaler.update()
    if np.isnan(A_to_B_losses["loss"]).any(): 
        print("NaN values detected in loss history. Check earlier stages.")

    #6: print out total time 
    print("Total time per sample is: "+str(time.time()-total_time))

    return A_to_B_losses, model, optimizer, scaler

def diffusion_sampling(condition, model):
    # TODO make comparision betweenn ddim and p_sampke
    sampled_images = test_diffusion.ddim_sample_loop_inpaint(
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
                voided_mri = batch["voided_mri_full_res"].reshape(-1, 1, 240, 240, 155)
                healthy_region_mask_corrected = batch["healthy_region_mask_corrected_full_res"].reshape(-1, 1, 240, 240, 155)
                target = batch["mri"].reshape(-1, 1, 240, 240, 155) # The original MRI scan

                print(f"healthy_region_mask_corrected: {healthy_region_mask_corrected.shape}")
                print(f"voided_mri: {voided_mri.shape}")
                original_mean, original_std = batch['mean'], batch['std']

                condition = torch.cat(
                    (voided_mri, healthy_region_mask_corrected),
                    dim=1
                ).to(args.device)
                target = target.to(args.device)
                
                # Save MRI
                mri_image_batch = target[0][0].cpu().numpy()  
                mri_image_batch = mri_image_batch * original_std + original_mean
                # Rescale the values from z-score to original
                nib.save(nib.Nifti1Image(
                        mri_image_batch,
                        affine=np.eye(4)),
                        f'{args.path_checkpoint}/scans/{epoch}_{i}_original.nii.gz'
                    )
                # Voided MRI
                mri_image_batch = voided_mri[0][0].cpu().numpy()  
                mri_image_batch = mri_image_batch * original_std + original_mean
                # Rescale the values from z-score to original
                nib.save(nib.Nifti1Image(
                        mri_image_batch,
                        affine=np.eye(4)),
                        f'{args.path_checkpoint}/scans/{epoch}_{i}_voided_mri.nii.gz'
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
                nii_pred_image = predict_ct_image_batch * original_std + original_mean
                
                nib.save(
                    nib.Nifti1Image(nii_pred_image, affine=np.eye(4)), 
                    f'{args.path_checkpoint}/scans/{epoch}_{i}_validation.nii.gz'
                    )
            
                if i >= 0: # Uses only 3 cases for validation
                    break
    with torch.no_grad():
        pbar = tqdm(val_loader_mri, desc="Validation")
        for i, batch in enumerate(pbar):
            voided_mri = batch["voided_mri"].reshape(-1, 1, *args.patch_size).to(args.device)
            healthy_region_mask_corrected = batch["healthy_region_mask_corrected"].reshape(-1, 1, *args.patch_size).to(args.device)
            target = batch["target"].reshape(-1, 1, *args.patch_size).to(args.device) # Regions healthy

            # Start and end of the bounding box
            start, end = crop_foreground.compute_bounding_box(img=healthy_region_mask_corrected[0].cpu().numpy())
            start, end = check_and_clamp_bounding_box_shape(start, end, args.patch_size, min_dim=12)

            condition = torch.cat(
                    (voided_mri, healthy_region_mask_corrected),
                    dim=1
                ).to(args.device)
            target = target.to(args.device)
            
            if args.timestep_respacing_val=='':
                print("Using random steps t for validation.")
                # Predict patch
                t_all = []
                for _t_ in range(6):
                    t, weights = schedule_sampler.sample(condition.shape[0], args.device)
                    t_all.append(t)
            else:
            
                # Select timesteps for better validation
                # Fixed tensors
                t1 = torch.tensor([1], device=args.device)

                # Random tensors within specified ranges
                t2 = torch.randint(2, 6, (1,), device=args.device)   # 2 to 5
                t3 = torch.randint(5, 11, (1,), device=args.device)  # 5 to 10
                t4 = torch.randint(10, 16, (1,), device=args.device) # 10 to 15
                t5 = torch.randint(15, 21, (1,), device=args.device) # 15 to 20
                t6 = torch.randint(20, 25, (1,), device=args.device) # 20 to 24
                t_all = [t1,t2,t3,t4,t5,t6]
            for t in t_all:
                _, _, sampled_images = train_diffusion.training_losses(A_to_B_model, target, condition, t, train_metric=args.train_metric)
                
                # Computed only on the region to inpaint
                # Normalisation not needed
                num_active_elements = torch.sum(healthy_region_mask_corrected.float())
                # MAE
                MAE_value = eval_MAE(sampled_images, target)
                MAE_value = MAE_value * healthy_region_mask_corrected.float()
                MAE_value = torch.sum(MAE_value)
                MAE_value = MAE_value / num_active_elements
                # MSE
                MSE_value = eval_MSE(sampled_images, target)
                MSE_value = MSE_value * healthy_region_mask_corrected.float()
                MSE_value = torch.sum(MSE_value)
                MSE_value = MSE_value / num_active_elements
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
                # Crop to focus more on the region that was inpainted
                sampled_images_here = sampled_images_here[:,:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                target_here = target_here[:,:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]

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

        scaler = torch.amp.GradScaler("cuda")# init_scale=1024.0) # resumed from epoch 40 due to nan output

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
    parser.add_argument('--intensity_scale_range', action='store_true', help='Linearly normalize the intensities to -1 and 1 (from the z-score).')
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
    
    if args.intensity_scale_range:
        name += "_"+"ranged"

    if args.prob != 0:
        name += f"_DA_{args.prob}"

    name += f"_{args.noise_schedule}"
    
        
    args.path_checkpoint = join(args.path_checkpoint, name)
    wandb.init(project="BraTS2025_inpaint", name=name, dir=args.path_checkpoint)
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
        print(f"Using intensity ranged: {args.intensity_scale_range}")
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
        min_mask_voxels=4,
        intensity_scale_range=args.intensity_scale_range,
        json_file='/homes/andre.ferreira/BraTS2025/Dataset/Local-Synthesis/data_split.json')

    
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
            if args.intensity_scale_range:
                args.global_min = -1 # Because it's normalised between -1 and 1
                args.global_max = 1
            else:
                args.global_min = -6 # 6 STDs
                args.global_max = 6 
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

    scaler = torch.amp.GradScaler("cuda")# , init_scale=1024.0) # resumed from epoch 40 due to nan output
    
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
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Retrieve the epoch and best loss
        begin_epoch = checkpoint['epoch'] 
        best_loss = checkpoint['best_loss'] 
        best_psnr = checkpoint['best_psnr']  
        #best_loss = 10000
        #begin_epoch = 0 
        #best_psnr=0 
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
            print(f'New validation best 🎉 Save the latest best model. smoothed_val_mae={smoothed_val_losses} | Real mean_mae={mean_mae}')
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
    