import nrrd
import os
from PIL import Image
import numpy as np

# Directory containing the NRRD files
nrrd_directory = "./RESULTS/size64_steps300_classes9_s1-5/history_samples"

# Directory to save the PNG files
png_directory = "./RESULTS/size64_steps300_classes9_s1-5/pngs"
os.makedirs(png_directory, exist_ok=True)

# Index of the slice to extract
slice_index = 35  # Modify this according to the slice you need

# Loop through each NRRD file in the directory
for filename in os.listdir(nrrd_directory):
    if filename.endswith(".nrrd"):
        # Construct the full file path
        file_path = os.path.join(nrrd_directory, filename)

        # Read the NRRD file
        data, header = nrrd.read(file_path)

        # Extract the specified slice
        # Assuming the slices are along the third dimension
        slice_data = data[:, slice_index, :]
        normalized_slice_data = ((slice_data + 1) / 2) * 255

        # Convert to an image (scale values if necessary)
        image = Image.fromarray(normalized_slice_data.astype(np.uint8))

        # Save the image as PNG with the same name as the NRRD file
        png_filename = os.path.splitext(filename)[0] + '_v2.png'
        png_path = os.path.join(png_directory, png_filename)
        image.save(png_path)

print("PNG files saved successfully.")
