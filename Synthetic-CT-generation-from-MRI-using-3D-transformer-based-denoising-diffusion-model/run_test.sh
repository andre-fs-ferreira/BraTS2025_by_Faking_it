#!/bin/bash

#SBATCH --partition=GPUampere
#SBATCH --time=720:00:00
#SBATCH --job-name=test_BRATS_mc_25_2000_IDDPM_HN_TH_AB_128_128_32_last_clip
#SBATCH --output=sbatch_out/test_BRATS_mc_25_2000_IDDPM_HN_TH_AB_128_128_32_last_clip_%J.txt
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Read arguments from command line
START_CASE=$1
END_CASE=$2

python test_infer_mc_IDDPM.py \
    --network SwinVIT \
    --num_workers 1 \
    --patch_size 128 128 32 \
    --cache_rate 1.0 \
    --train_metric MSE \
    --timestep_respacing "25" \
    --timestep_respacing_val "25" \
    --sw_batch_size 1 \
    --overlap 0.5 \
    --overlap_mode 'constant' \
    --verbose \
    --mode test \
    --json_file /homes/andre.ferreira/BraTS2025/Dataset/Local-Synthesis/data_split_test.json \
    --prediction_path /homes/andre.ferreira/BraTS2025/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/predictions \
    --path_checkpoint /homes/andre.ferreira/BraTS2025/logs/MC-IDDPM_2_2000_timestep_25_patchsize_2_SwinVIT_MSE_MAE_SSIM_128_128_32_ranged_DA_0.5/wandb/run-20250715_104911-sb2ubfms/files/model/A_to_B_model_latest.pt \
    --intensity_scale_range \
    --start_case "$START_CASE" \
    --end_case "$END_CASE" \
    --clip_denoised

python test_infer_mc_IDDPM.py \
    --network SwinVIT \
    --num_workers 1 \
    --patch_size 128 128 32 \
    --cache_rate 1.0 \
    --train_metric MSE \
    --timestep_respacing "ddim25" \
    --timestep_respacing_val "ddim25" \
    --sw_batch_size 1 \
    --overlap 0.5 \
    --overlap_mode 'constant' \
    --verbose \
    --mode test \
    --json_file /homes/andre.ferreira/BraTS2025/Dataset/Local-Synthesis/data_split_test.json \
    --prediction_path /homes/andre.ferreira/BraTS2025/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/predictions \
    --path_checkpoint /homes/andre.ferreira/BraTS2025/logs/MC-IDDPM_2_2000_timestep_25_patchsize_2_SwinVIT_MSE_MAE_SSIM_128_128_32_ranged_DA_0.5/wandb/run-20250715_104911-sb2ubfms/files/model/A_to_B_model_latest.pt \
    --intensity_scale_range \
    --start_case "$START_CASE" \
    --end_case "$END_CASE" \
    --clip_denoised

#python test_infer_mc_IDDPM.py \
#    --network SwinVIT \
#    --num_workers 8 \
#    --patch_size 128 128 32 \
#    --cache_rate 1.0 \
#    --train_metric MSE \
#    --timestep_respacing "inpaint25" \
#    --timestep_respacing_val "inpaint25" \
#    --sw_batch_size 1 \
#    --overlap 0.5 \
#    --overlap_mode 'constant' \
#    --verbose \
#    --mode test \
#    --json_file /homes/andre.ferreira/BraTS2025/Dataset/Local-Synthesis/data_split_test.json \
#    --prediction_path /homes/andre.ferreira/BraTS2025/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/predictions \
#    --path_checkpoint /homes/andre.ferreira/BraTS2025/logs/MC-IDDPM_2_1000_timestep_25_patchsize_2_SwinVIT_MSE_MAE_SSIM_128_128_32_DA_0.5/wandb/run-20250704_092359-eghxu232/files/model/A_to_B_model_latest.pt
