#!/bin/bash

#SBATCH --partition=GPUampere
#SBATCH --time=720:00:00
#SBATCH --job-name=train_BRATS_Syn_mc_IDDPM_HN_TH_AB_128_128_32_global_1000steps
#SBATCH --output=sbatch_out/train_BRATS_Syn_mc_IDDPM_HN_TH_AB_128_128_32_global_1000steps_%J.txt
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Nice shape to use -> 128 128 128 (all tumours are smaller than this value :D)
# Training script for Synthetic-CT generation from MRI
# --network SwinVIT, SwinUNETR_vit, SwinUNETR, nnUNet

python train_mc_IDDPM_BraSyn.py \
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
    --path_checkpoint /homes/andre.ferreira/BraTS2025/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model_BraSys/logs \
    --n_epochs 3000 \
    --val_interval 3000 \
    --pacience 500 \
    --add_train_metric MAE SSIM \
    --add_train_metric_weight 1 1 \
    --prob 0.5 \
    --verbose \
    --shuffle \
    --use_global_stats \
    --lr 0.00002 \
    --noise_schedule cosine
    #--dataset_path /projects/nian/synthrad2025/Dataset/ \
    #--use_cosine_scheduler
    #--resume /homes/andre.ferreira/BraTS2025/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model_BraSys/logs/MC-IDDPM_2_1000_timestep_25_patchsize_2_SwinVIT_MSE_MAE_SSIM_128_128_32_DA_0.5/wandb/run-20250708_144337-rnmwcaqu/files/model/A_to_B_model_latest.pt

