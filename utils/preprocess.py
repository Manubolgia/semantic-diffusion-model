import os
import nrrd
import numpy as np
import argparse

def read_nrrd(file_path):
    data, header = nrrd.read(file_path)
    return data

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

def _list_nrrd_files_recursively(data_dir):
    results = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".nrrd"):
                full_path = os.path.join(root, file)
                results.append(full_path)
    return results

def find_smallest_depth(cta_files):
    smallest_depth = float('inf')
    for file in cta_files:
        data = read_nrrd(file)
        depth = data.shape[2] 
        if depth < smallest_depth:
            smallest_depth = depth
    return smallest_depth

def save_cropped_image(output_dir_base, original_path, cropped_data):
    # Determine if the file is from the training or validation set
    set_type = 'training' if 'training' in original_path else 'validation'

    # Construct new path
    new_filename = os.path.basename(original_path)
    new_path = os.path.join(output_dir_base, set_type, new_filename)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Save the cropped image
    nrrd.write(new_path, cropped_data)

def calculate_final_size(original_size):
    # Calculate final size using the provided formula
    return int(np.floor(original_size * (1 - (np.sqrt(2) - 1) * 0.5 * np.sqrt(2))))


def compute_global_mean_std(cta_files):
    mean = 0.0
    M2 = 0.0
    pixel_count = 0

    for file in cta_files:
        data = read_nrrd(file)
        flattened_data = data.flatten()
        pixel_count += flattened_data.size
        for pixel in flattened_data:
            delta = pixel - mean
            mean += delta / pixel_count
            delta2 = pixel - mean
            M2 += delta * delta2

    variance = M2 / pixel_count
    std_dev = np.sqrt(variance)
    return mean, std_dev

def normalize_dataset(cta_files, mean, std):
    for file in cta_files:
        data = read_nrrd(file)
        normalized_data = normalize_image(data, mean, std)
        nrrd.write(file, normalized_data)

def normalize_image(image, mean, std):
    return (image - mean) / std

def process_images(data_dir, normalize=False):
    cta_source_dir = os.path.join(data_dir, 'CTCA')
    annotation_source_dir = os.path.join(data_dir, 'Annotations')

    cta_destination_dir = os.path.join(data_dir, 'cta')
    annotation_destination_dir = os.path.join(data_dir, 'annotation')

    cta_files = _list_nrrd_files_recursively(cta_source_dir)
    annotation_files = _list_nrrd_files_recursively(annotation_source_dir)

    smallest_depth = find_smallest_depth(cta_files)

    first_image = read_nrrd(cta_files[0])
    original_size = first_image.shape[1]  # Assuming the images are square (HxW)
    final_size = calculate_final_size(original_size)
    print("Original size detected: ", original_size)
    print("Calculated final size for cropping: ", final_size)

    if not cta_files:
        print("No CTA files found.")
        return

    # Phase 1: Cropping
    for cta_path, annotation_path in zip(cta_files, annotation_files):
        nrrd_image = read_nrrd(cta_path)
        nrrd_class = read_nrrd(annotation_path)

        print(f"Original dimensions of {os.path.basename(cta_path)}: {nrrd_image.shape}")
        print(f"Original dimensions of {os.path.basename(annotation_path)}: {nrrd_class.shape}")

        cropped_image, cropped_class = center_crop_arr([nrrd_image, nrrd_class], final_size, smallest_depth)

        print(f"Cropped dimensions of {os.path.basename(cta_path)}: {cropped_image.shape}")
        print(f"Cropped dimensions of {os.path.basename(annotation_path)}: {cropped_class.shape}")

        save_cropped_image(cta_destination_dir, cta_path, cropped_image)
        save_cropped_image(annotation_destination_dir, annotation_path, cropped_class)

    # Phase 2: Normalization (if enabled)
    if normalize:
        mean, std = compute_global_mean_std(cta_files)
        normalize_dataset(cta_files, mean, std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Cropping and Normalization Script")

    parser.add_argument("--data_folder", required=True, help="Path to the preprocessed data folder")
    parser.add_argument("--normalize", type=bool, default=False, help="Enable normalization (default: False)")

    args = parser.parse_args()
    
    process_images(args.data_folder, args.normalize)

