import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           binary_fill_holes)
from scipy.ndimage.morphology import generate_binary_structure

def process_nrrd_files(src_dir, dest_dir):
    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Traverse the directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.nrrd'):
                # Construct the file path
                file_path = os.path.join(root, file)

                # Read the NRRD file
                image = sitk.ReadImage(file_path)
                array = sitk.GetArrayFromImage(image)

                # Get unique labels (excluding 0 if it represents background)
                unique_labels = np.unique(array[array != 0])

                # Initialize an empty array for the processed label map
                processed_array = np.zeros_like(array)

                # Structuring element for morphological operations
                struct = generate_binary_structure(3, 1)  # 3D structuring element

                # Apply morphological operations to each label
                for label in unique_labels:
                    binary_image = (array == label)
                    # Apply closing
                    closed_binary_image = binary_closing(binary_image, structure=struct)
                    # Apply dilation followed by erosion
                    dilated_image = binary_dilation(closed_binary_image, structure=struct)
                    eroded_image = binary_erosion(dilated_image, structure=struct)
                    # Apply fill holes
                    filled_image = binary_fill_holes(eroded_image)
                    processed_array[filled_image] = label


                # Convert array back to image
                processed_image = sitk.GetImageFromArray(processed_array)
                processed_image.CopyInformation(image)

                # Construct the destination path
                rel_path = os.path.relpath(root, src_dir)
                dest_path = os.path.join(dest_dir, rel_path)
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)

                # Save the processed file
                new_file_path = os.path.join(dest_path, file)
                sitk.WriteImage(processed_image, new_file_path)

def process_nifti_files(src_dir, dest_dir):
    """
    Process NIfTI files as done in process_nrrd_files()
    """
    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Traverse the directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                # Construct the file path
                file_path = os.path.join(root, file)

                # Read the NRRD file
                image = nib.load(file_path)
                array = image.get_fdata()

                # Get unique labels (excluding 0 if it represents background)
                unique_labels = np.unique(array[array != 0])

                # Initialize an empty array for the processed label map
                processed_array = np.zeros_like(array)

                # Structuring element for morphological operations
                struct = generate_binary_structure(3, 3)  # 3D structuring element

                # Apply morphological operations to each label
                for label in unique_labels:
                    if label != 8:  # Exclude label 8
                        binary_image = (array == label)
                        # Apply closing
                        closed_binary_image = binary_closing(binary_image, structure=struct)
                        # Apply dilation followed by erosion
                        dilated_image = binary_dilation(closed_binary_image, structure=struct)
                        eroded_image = binary_erosion(dilated_image, structure=struct)
                        # Apply fill holes
                        filled_image = binary_fill_holes(eroded_image)
                        processed_array[filled_image] = label


                # Convert array back to image
                processed_image = nib.Nifti1Image(processed_array, affine=image.affine, header=image.header)

                # Construct the destination path
                rel_path = os.path.relpath(root, src_dir)
                dest_path = os.path.join(dest_dir, rel_path)
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)

                # Save the processed file
                new_file_path = os.path.join(dest_path, file)
                nib.save(processed_image, new_file_path)

# Example usage
src_directory = '/home/data/farid/vessel_segmentation/kaggle_dataset/annotation_totalseg'  # Your source directory
dest_directory = '/home/data/farid/vessel_segmentation/kaggle_dataset/annotation'  # Your destination directory
dataset_mode = 'nifti'  # 'nrrd', 'nifti' or 'all'

if dataset_mode == 'nrrd':
    process_nrrd_files(src_directory, dest_directory)
elif dataset_mode == 'nifti':
    process_nifti_files(src_directory, dest_directory)
elif dataset_mode == 'all':
    process_nrrd_files(src_directory, dest_directory)
    process_nifti_files(src_directory, dest_directory)
