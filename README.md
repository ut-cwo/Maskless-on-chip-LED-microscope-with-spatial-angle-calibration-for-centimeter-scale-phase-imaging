ensfree Tissue Section Reconstruction Code

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


Do NOT place the Utility folder or the MATLAB code inside the dataset folder.
Keep them in the same working directory and update the path in your reconstruction script as needed.

How to Run the Reconstruction

Open MATLAB.

Navigate to your working directory (where reconstruct.m is located).

Edit the script to set the correct dataset path if needed.

Run:

reconstruct


The script loads the raw measurements from the dataset folder, applies the calibration data, and performs the iterative reconstruction to produce amplitude and phase images.
