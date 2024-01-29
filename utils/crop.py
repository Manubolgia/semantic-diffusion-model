import os
import blobfile as bf
import nrrd
import nibabel as nib
import numpy as np
import argparse

def _list_nrrd_files_recursively(data_dir):
    """""
    List all nrrd files recursively
        
    """""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() == "nrrd":
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_nrrd_files_recursively(full_path))
    return results

def _list_nifti_files_recursively(data_dir):
    """""
    List all nifti files recursively
    
    """""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() == "gz":
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_nifti_files_recursively(full_path))
    return results

def _list_all_files_recursively(data_dir):
    """""
    List all nrrd and nifti files recursively

    """""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and (ext.lower() == "nrrd" or ext.lower() == "gz"):
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_all_files_recursively(full_path))
    return results


def read_nrrd(file_path):
    """
    Read nrrd file and return numpy array
    """
    data, header = nrrd.read(file_path)
    return data

def read_nifti(file_path):
    """
    Read nifti file and return numpy array, analog to read_nrrd
    """
    data = nib.load(file_path).get_fdata()
    return data

def read_file(file_path):
    """
    Handle the logic of calling read_nrrd or read_nifti according to the file extension
    """
    if file_path.endswith('.nrrd'):
        return read_nrrd(file_path)
    elif file_path.endswith('.nii.gz'):
        return read_nifti(file_path)
    else:
        raise ValueError(f"Invalid file extension: {file_path}")
    


def center_crop_arr(np_list, crop_dims, crop_D):
    np_image, np_class = np_list

    H, W, D = np_image.shape
    crop_H, crop_W = crop_dims, crop_dims

    # Ensure the crop size is smaller than the image size
    crop_D = min(crop_D, D)
    crop_H = min(crop_H, H)
    crop_W = min(crop_W, W)

    start_D = (D - crop_D) // 2
    start_H = (H - crop_H) // 2
    start_W = (W - crop_W) // 2

    end_D = start_D + crop_D
    end_H = start_H + crop_H
    end_W = start_W + crop_W

    cropped_image = np_image[start_H:end_H, start_W:end_W, start_D:end_D]
    cropped_class = np_class[start_H:end_H, start_W:end_W, start_D:end_D]

    return cropped_image, cropped_class


def find_smallest_depth(cta_files):
    smallest_depth = float('inf')
    for file in cta_files:
        data = read_file(file)
        depth = data.shape[2] 
        if depth < smallest_depth:
            smallest_depth = depth
    return smallest_depth

def save_cropped_image(output_dir_base, original_path, cropped_data, dataset_mode='all'):
    # Determine if the file is from the training or validation set
    set_type = 'training' if 'training' in original_path else 'validation'

    # Construct new path
    new_filename = os.path.basename(original_path)
    new_path = os.path.join(output_dir_base, set_type, new_filename)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Save the cropped image
    if dataset_mode == 'nifti':
        nib.save(nib.Nifti1Image(cropped_data, affine=np.eye(4)), new_path)
    elif dataset_mode == 'nrrd':
        nrrd.write(new_path, cropped_data)
    elif dataset_mode == 'all':
        if original_path.endswith('.nrrd'):
            nrrd.write(new_path, cropped_data)
        elif original_path.endswith('.nii.gz'):
            nib.save(nib.Nifti1Image(cropped_data, affine=np.eye(4)), new_path)
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

def calculate_final_size(original_size):
    # Calculate final size using the provided formula
    return int(np.floor(original_size * (1 - (np.sqrt(2) - 1) * 0.5 * np.sqrt(2))))


def process_images(data_dir, dataset_mode='all'):
    cta_source_dir = os.path.join(data_dir, 'CTCA')
    annotation_source_dir = os.path.join(data_dir, 'Annotations')

    cta_destination_dir = os.path.join(data_dir, 'cta')
    annotation_destination_dir = os.path.join(data_dir, 'annotation')

    if dataset_mode == 'nifti':
        cta_files = _list_nifti_files_recursively(cta_source_dir)
        annotation_files = _list_nifti_files_recursively(annotation_source_dir)
    elif dataset_mode == 'nrrd':
        cta_files = _list_nrrd_files_recursively(cta_source_dir)
        annotation_files = _list_nrrd_files_recursively(annotation_source_dir)
    elif dataset_mode == 'all':
        cta_files = _list_all_files_recursively(cta_source_dir)
        annotation_files = _list_all_files_recursively(annotation_source_dir)
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

    smallest_depth = find_smallest_depth(cta_files)

    # Calculate final size but loading according to the dataset mode

    if dataset_mode == 'nifti':
        first_image = read_nifti(cta_files[0])
    elif dataset_mode == 'nrrd':
        first_image = read_nrrd(cta_files[0])
    elif dataset_mode == 'all':
        first_image = read_file(cta_files[0])
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")
    
    original_size = first_image.shape[1]  # Assuming the images are square (HxW)
    final_size = calculate_final_size(original_size)
    print("Original size detected: ", original_size)
    print("Calculated final size for cropping: ", final_size)

    if not cta_files:
        print("No CTA files found.")
        return

    # Cropping
    for cta_path, annotation_path in zip(cta_files, annotation_files):
        if dataset_mode == 'nifti':
            ct_image = read_nifti(cta_path)
            ct_class = read_nifti(annotation_path)
        elif dataset_mode == 'nrrd':
            ct_image = read_nrrd(cta_path)
            ct_class = read_nrrd(annotation_path)
        elif dataset_mode == 'all':
            ct_image = read_file(cta_path)
            ct_class = read_file(annotation_path)
        else:
            raise ValueError(f"Invalid dataset mode: {dataset_mode}")

        print(f"Original dimensions of {os.path.basename(cta_path)}: {ct_image.shape}")
        print(f"Original dimensions of {os.path.basename(annotation_path)}: {ct_class.shape}")

        cropped_image, cropped_class = center_crop_arr([ct_image, ct_class], final_size, smallest_depth)

        print(f"Cropped dimensions of {os.path.basename(cta_path)}: {cropped_image.shape}")
        print(f"Cropped dimensions of {os.path.basename(annotation_path)}: {cropped_class.shape}")

        save_cropped_image(cta_destination_dir, cta_path, cropped_image, dataset_mode)
        save_cropped_image(annotation_destination_dir, annotation_path, cropped_class, dataset_mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Cropping Script")

    parser.add_argument("--data_folder", required=True, help="Path to the preprocessed data folder")
    parser.add_argument("--dataset_mode", type=str, default="all", help="Dataset mode (default: all)")

    args = parser.parse_args()
    
    process_images(args.data_folder, args.dataset_mode)

