#!/bin/bash

#SBATCH --partition=GPUampere
#SBATCH --time=720:00:00
#SBATCH --job-name=train_BRATS_mc_IDDPM_HN_TH_AB_128_128_32_1000stepslinear
#SBATCH --output=sbatch_out/train_BRATS_mc_IDDPM_HN_TH_AB_128_128_32_1000stepslinear_%J.txt
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Nice shape to use -> 128 128 128 (all tumours are smaller than this value :D)
# Training script for Synthetic-CT generation from MRI
# --network SwinVIT, SwinUNETR_vit, SwinUNETR, nnUNet

python train_mc_IDDPM.py \
    --network SwinVIT \
    --batch_size_train 2 \
    --patch_num 2 \
    --num_workers 8 \
    --patch_size 128 128 32 \
    --cache_rate 1.0 \
    --train_metric MSE \
    --timestep_respacing "" \
    --timestep_respacing_val "" \
    --sw_batch_size 16 \
    --overlap 0.5 \
    --overlap_mode 'constant' \
    --path_checkpoint /homes/andre.ferreira/BraTS2025/logs \
    --n_epochs 3000 \
    --val_interval 3000 \
    --pacience 500 \
    --add_train_metric MAE SSIM \
    --add_train_metric_weight 1 1 \
    --prob 0.5 \
    --verbose \
    --shuffle \
    --intensity_scale_range \
    --lr 0.00002 \
    --noise_schedule linear \
    --resume /homes/andre.ferreira/BraTS2025/logs/MC-IDDPM_2_3000_timestep__patchsize_2_SwinVIT_MSE_MAE_SSIM_128_128_32_ranged_DA_0.5_linear/wandb/run-20250721_081538-1p27scyt/files/model/A_to_B_model_best.pt
    #--dataset_path /projects/nian/synthrad2025/Dataset/ \
    #--use_cosine_scheduler
    #

# Training from pre-trained model    
# python train_mc_IDDPM.py \
#     --network nnUNet \
#     --batch_size_train 1 \
#     --patch_num 1 \
#     --num_workers 8 \
#     --patch_size 96 96 96 \
#     --dataset_path /projects/nian/synthrad2025/Dataset/ \
#     --region HN TH AB \
#     --cache_rate 0 \
#     --task Task1 \
#     --timestep_respacing 1000 \
#     --timestep_respacing_val 50 \
#     --sw_batch_size 2 \
#     --overlap 0.5 \
#     --path_checkpoint ../../results/MC-IDDPM \
#     --n_epochs 5000 \
#     --val_interval 50 \
#     --load_pretrained \
#     --path_pretrained /projects/nian/synthrad2025/src/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/network/pre_trained/TotalSegmentator/mri/Dataset852_TotalSegMRI_total_3mm_1088subj/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth \
#     --freeze_epochs 10 \
#     --verbose
