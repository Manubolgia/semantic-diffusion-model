import nrrd
import nibabel as nib
import os
from PIL import Image
import numpy as np

dataset_mode = 'nifti' #'nifti', 'nrrd'
# Directory containing the NRRD files
file_directory = "./RESULTS/size64_steps300_classes9_s1-5/history_samples"

# Directory to save the PNG files
png_directory = "./RESULTS/size64_steps300_classes9_s1-5/pngs"
os.makedirs(png_directory, exist_ok=True)

# Index of the slice to extract
slice_index = 35  # Modify this according to the slice you need

if dataset_mode == 'nrrd':
    # Loop through each NRRD file in the directory
    for filename in os.listdir(file_directory):
        if filename.endswith(".nrrd"):
            # Construct the full file path
            file_path = os.path.join(file_directory, filename)

            # Read the NRRD file
            data, header = nrrd.read(file_path)

            # Extract the specified slice
            # Assuming the slices are along the third dimension
            slice_data = data[:, slice_index, :]
            normalized_slice_data = ((slice_data + 1) / 2) * 255

            # Convert to an image (scale values if necessary)
            image = Image.fromarray(normalized_slice_data.astype(np.uint8))

            # Save the image as PNG with the same name as the NRRD file
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(png_directory, png_filename)
            image.save(png_path)
elif dataset_mode == 'nifti':
    # Loop through each NRRD file in the directory
    for filename in os.listdir(file_directory):
        if filename.endswith(".nii.gz"):
            # Construct the full file path
            file_path = os.path.join(file_directory, filename)

            # Read the NRRD file
            data, header = nib.load(file_path)

            # Extract the specified slice
            # Assuming the slices are along the third dimension
            slice_data = data[:, slice_index, :]
            normalized_slice_data = ((slice_data + 1) / 2) * 255

            # Convert to an image (scale values if necessary)
            image = Image.fromarray(normalized_slice_data.astype(np.uint8))
            # Save the image as PNG with the same name as the nifti file
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(png_directory, png_filename)
            image.save(png_path)
else:
    raise ValueError(f"Invalid dataset mode: {dataset_mode}")
print("PNG files saved successfully.")
