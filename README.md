# Lensfree Tissue Section Reconstruction Code

This repository contains the MATLAB and Python (JAX) code used to reconstruct quantitative phase images from raw multi-angle lensfree microscopy datasets described in our manuscript.

Please cite our manuscript if you use this code (details to be added).

---

## Dataset Download

The raw multi-angle measurement datasets for thin tissue sections (mouse kidney, ovary, jejunum, cuboidal epithelium, etc.) are hosted on the Texas Data Repository:

https://dataverse.tdl.org/dataverse/on_chip_microscopy_with_spatially_varying_angle_calibration

Download the dataset folder(s) you need and place them anywhere on your computer.

---

## Folder Structure

Your working directory should look like:

Your_Working_Directory/
Dataset_Folder/
patch_1.mat
patch_2.mat
...
Utility/
core_functions/
unlocbox/
reconstruct.m
Jax_code.py


Do not place the `Utility` folder inside the dataset folder.  
Keep the reconstruction code and dataset files separate.

---

## How to Run the MATLAB Reconstruction

Open MATLAB.

Navigate to your working directory.

Edit `reconstruct.m` to point to your dataset folder.

Run:

reconstruct.m

The script loads the raw measurements, applies the LED calibration, and performs the iterative reconstruction to generate amplitude and phase images.

---

## JAX / Python Reconstruction Code

A GPU-accelerated Python implementation using JAX is also provided.  
It reconstructs the same multi-angle datasets using:

- forward model evaluation  
- gradient-based updates  
- total variation regularization  
- Wiener filtering (optional)  

The script expects the same `patch_*.mat` dataset files and uses the same folder structure as MATLAB.

---

## WSL Path Setup (Windows Users)

If running Python under WSL, convert Windows paths such as:

`D:\Manuscripts\Lensless_2D\Dataset\QPI_phase_target`

to the WSL format:

`/mnt/d/Manuscripts/Lensless_2D/Dataset/QPI_phase_target`

Set this as `wsl_base` inside the Python script.

---

## How to Run the JAX Reconstruction

Install required Python packages:

pip install jax jaxlib numpy scipy matplotlib scikit-image pillow

Set the following inside `Jax_code.py`:

- `wsl_base` → path to your dataset directory  
- `mat_file` → patch file to reconstruct  

Run:

python Jax_code.py

The script loads the raw measurements and calibration data, performs iterative reconstruction on the GPU, and saves results to:

Recon_python.mat

for comparison or further analysis in MATLAB.

---
