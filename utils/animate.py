import nrrd
import nibabel as nib
import os
from PIL import Image
import numpy as np

dataset_mode = 'nifti' #'nifti', 'nrrd'
# Directory containing the files
#file_directory = "..."

# Directory to save the PNG files
png_directory = "..."
os.makedirs(png_directory, exist_ok=True)

# Index of the slice to extract
slice_index = 30  # Modify this according to the slice you need

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
            slice_data = data[:, :, slice_index]
            normalized_slice_data = ((slice_data + 1) / 2) * 255

            # Convert to an image (scale values if necessary)
            image = Image.fromarray(normalized_slice_data.astype(np.uint16))

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
            data = nib.load(file_path).get_fdata()

            # Extract the specified slice
            # Assuming the slices are along the third dimension
            slice_data = data[:,:,slice_index]
            normalized_slice_data = ((slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))) * 255

            # Convert to an image (scale values if necessary)
            image = Image.fromarray(normalized_slice_data.astype(np.uint8))
            # Save the image as PNG with the same name as the nifti file
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(png_directory, png_filename)
            image.save(png_path)
else:
    raise ValueError(f"Invalid dataset mode: {dataset_mode}")
print("PNG files saved successfully.")
