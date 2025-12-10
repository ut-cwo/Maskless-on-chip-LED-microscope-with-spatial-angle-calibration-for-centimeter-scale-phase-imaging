Lensfree Tissue Section Reconstruction Code

This repository contains the MATLAB code and utility functions used to reconstruct quantitative phase images from the raw multi-angle lensfree microscopy datasets described in our manuscript.

Dataset Download

The raw multi-angle measurement datasets for all thin tissue sections (mouse kidney, ovary, jejunum, cuboidal epithelium, etc.) are hosted on the Texas Data Repository:

🔗 https://dataverse.tdl.org/dataverse/on_chip_microscopy_with_spatially_varying_angle_calibration

Download whichever tissue-section dataset folders you need and place them anywhere on your local machine.

Folder Structure

Your setup should look like:

Your_Working_Directory/
    Dataset_Folder/           % downloaded from Dataverse
        patch_1.mat
        patch_2.mat
        ...
    Utility/
        core_functions/
        unlocbox/
    reconstruct.m


Do not place the Utility folder or the MATLAB code inside the dataset folder.
Keep them in the same working directory and update the path in your reconstruction script as needed.

How to Run the MATLAB Reconstruction

Open MATLAB.

Navigate to your working directory (where reconstruct.m is located).

Edit the script to point to your dataset folder.

Run:

reconstruct


The script loads the raw measurements, applies the calibration data, and performs the iterative reconstruction to produce amplitude and phase images.

JAX / Python Reconstruction Code

This repository also contains a JAX-based Python implementation of the reconstruction algorithm. It uses the same raw multi-angle datasets (patch_*.mat files) and calibration data, but runs the forward model, gradient updates, TV regularization, and Wiener filtering on the GPU using JAX.

The Python script expects the .mat files downloaded from the Texas Data Repository and uses the same folder structure shown above.

WSL Path Setup

When running under WSL on Windows, convert Windows paths such as:

D:\Manuscripts\Lensless_2D\Dataset\QPI_phase_target


to the WSL form:

/mnt/d/Manuscripts/Lensless_2D/Dataset/QPI_phase_target


Set this path as wsl_base inside the Python script.

Example Layout for MATLAB + JAX
Your_Working_Directory/
    Dataset_Folder/
        patch_1.mat
        patch_2.mat
        ...
    Utility/
        core_functions/
        unlocbox/
    reconstruct.m            % MATLAB version
    Jax_code.py         % JAX / Python version

How to Run the JAX Reconstruction

Install the required Python packages:
jax, jaxlib, numpy, scipy, matplotlib, scikit-image, pillow.

Open the Python script (e.g., Jax_code.py).

Set wsl_base and mat_file so they point to your dataset folder and the desired patch_*.mat file.

Run:

python Jax_code.py


The JAX reconstruction script loads the raw measurements and calibration data, performs the iterative reconstruction on the GPU, and saves the reconstructed phase and intermediate results to Recon_python.mat for analysis or comparison in MATLAB.
