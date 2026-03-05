#!/bin/bash

#SBATCH --partition=GPUampere
#SBATCH --time=720:00:00
#SBATCH --job-name=test_BRATS_mc_25_steps_IDDPM_HN_TH_AB_128_128_32_global_stats_latest_clip
#SBATCH --output=sbatch_out/test_BRATS_mc_25_steps_IDDPM_HN_TH_AB_128_128_32_global_stats_latest_clip_%J.txt
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
    --mode val \
    --data_dir /homes/andre.ferreira/BraTS2025/Dataset/Global-Synthesis_Train_missing \
    --json_file /homes/andre.ferreira/BraTS2025/Dataset/Global-Synthesis_Train/data_split.json \
    --prediction_path /homes/andre.ferreira/BraTS2025/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model_BraSys/predictions/Valves \
    --path_checkpoint /homes/andre.ferreira/BraTS2025/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model_BraSys/logs/MC-IDDPM_2_2000_timestep_25_patchsize_2_SwinVIT_MSE_MAE_SSIM_128_128_32_global_stats_DA_0.5/wandb/run-20250711_094606-8dl79m97/files/model/A_to_B_model_latest.pt \
    --use_global_stats \
    --start_case "$START_CASE" \
    --end_case "$END_CASE" \
    #--clip_denoised

#python test_infer_mc_IDDPM.py \
#    --network SwinVIT \
#    --num_workers 1 \
#    --patch_size 128 128 32 \
#    --cache_rate 1.0 \
#    --train_metric MSE \
#    --timestep_respacing "ddim25" \
#    --timestep_respacing_val "ddim25" \
#    --sw_batch_size 1 \
#    --overlap 0.5 \
#    --overlap_mode 'constant' \
#    --verbose \
#    --mode val \
#    --data_dir /homes/andre.ferreira/BraTS2025/Dataset/Global-Synthesis_Val_Local_missing \
#    --json_file /homes/andre.ferreira/BraTS2025/Dataset/Global-Synthesis_Train/data_split.json \
#    --prediction_path /homes/andre.ferreira/BraTS2025/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model_BraSys/predictions \
#    --path_checkpoint /homes/andre.ferreira/BraTS2025/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model_BraSys/logs/MC-IDDPM_2_2000_timestep_25_patchsize_2_SwinVIT_MSE_MAE_SSIM_128_128_32_global_stats_DA_0.5/wandb/run-20250711_094606-8dl79m97/files/model/A_to_B_model_latest.pt \
#    --use_global_stats \
#    --start_case "$START_CASE" \
#    --end_case "$END_CASE" \
#    --clip_denoised



