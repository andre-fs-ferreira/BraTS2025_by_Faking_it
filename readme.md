# BraTS 2025 Winning Solutions: Inpainting & Missing Modality

This repository contains the code used to train the winning solutions for the BraTS 2025 inpainting and missing modality tasks.

## 📁 Repository Structure

The codebase is divided by task. Inside each folder, you will find the dedicated shell scripts required for both training and testing (running inference).

* **`Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/`**
    * **Task:** Inpainting
* **`Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model_BraSys/`**
    * **Task:** Missing Modality

## ⚙️ Data Loading

All data loading logic is handled within `MonaiDataLoader.py`. Feel free to modify this file to adjust the data pipeline to your specific needs.

## 📄 Paper 

Our paper detailing the methodology will be available soon: 
> [**Achieving Over 10× Faster Sample Generation with Conditional Denoising Diffusion**](https://link.springer.com/book/9783032163646)

## ❓ Support

If you encounter any problems, bugs, or have questions about the code, please open a new issue in this repository.