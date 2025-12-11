Lensfree Tissue Section Reconstruction Code

This repository contains the MATLAB code and utility functions used to reconstruct quantitative phase images from the raw multi-angle lensfree microscopy datasets described in our manuscript.

Dataset Download

The raw multi-angle measurement datasets for thin tissue sections (mouse kidney, ovary, jejunum, cuboidal epithelium, etc.) are hosted on the Texas Data Repository:

https://dataverse.tdl.org/dataverse/on_chip_microscopy_with_spatially_varying_angle_calibration

Download the dataset folder(s) you need and place them anywhere on your computer.

Folder Structure

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

Do not place the Utility folder inside the dataset folder. Keep the reconstruction code and the datasets separate. Update paths inside reconstruct.m as needed.

How to Run the MATLAB Reconstruction

Open MATLAB.

Navigate to your working directory.

Edit reconstruct.m to point to your dataset folder.

Run:

reconstruct

The script loads the raw measurements, applies calibration, and performs the iterative reconstruction to generate amplitude and phase images.

JAX / Python Reconstruction Code

A Python implementation using JAX is also included. It reconstructs the same multi-angle datasets using GPU-accelerated forward models, gradient updates, TV regularization, and Wiener filtering.

It expects the same patch_*.mat files downloaded from the Texas Data Repository and uses the same folder structure as above.

WSL Path Setup

If running Python under WSL, convert Windows paths like:

D:\Manuscripts\Lensless_2D\Dataset\QPI_phase_target

to WSL paths:

/mnt/d/Manuscripts/Lensless_2D/Dataset/QPI_phase_target

Set this WSL path as wsl_base inside the Python script.

Example Directory Layout (MATLAB + JAX)

Your_Working_Directory/
  Dataset_Folder/
  Utility/
    core_functions/
    unlocbox/
  reconstruct.m
  Jax_code.py

How to Run the JAX Reconstruction

Install required Python packages:
jax, jaxlib, numpy, scipy, matplotlib, scikit-image, pillow

Open Jax_code.py

Set wsl_base and mat_file to your dataset folder and the desired patch_*.mat file

Run:

python Jax_code.py

The JAX script loads the raw measurements, applies calibration, performs iterative reconstruction on the GPU, and saves the final phase image and intermediate data to Recon_python.mat for further analysis or comparison in MATLAB.
