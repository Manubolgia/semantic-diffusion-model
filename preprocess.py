import os
import nrrd
import numpy as np
import argparse

def read_nrrd(file_path):
    data, header = nrrd.read(file_path)
    return data

def center_crop_arr(np_list, volume_size):

    np_image, np_class = np_list

    D, H, W = np_image.shape
    crop_D, crop_H, crop_W = [volume_size, volume_size, volume_size]

    # Ensure the crop size is smaller than the image size
    #crop_D = min(crop_D, D)
    crop_H = min(crop_H, H)
    crop_W = min(crop_W, W)

    start_D = (D - crop_D) // 2
    start_H = (H - crop_H) // 2
    start_W = (W - crop_W) // 2

    end_D = start_D + crop_D
    end_H = start_H + crop_H
    end_W = start_W + crop_W

    cropped_image = np_image[start_D:end_D, start_H:end_H, start_W:end_W]
    cropped_class = np_class[start_D:end_D, start_H:end_H, start_W:end_W]

    return cropped_image, cropped_class

def _list_nrrd_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() == "nrrd":
            results.append(full_path)
        #elif os.path.isdir(full_path):
            #results.extend(_list_nrrd_files_recursively(full_path))
    return results

def save_cropped_image(output_dir, original_path, cropped_data):
    # Construct new path
    new_filename = os.path.basename(original_path).replace('.nrrd', '_cropped.nrrd')
    new_path = os.path.join(output_dir, new_filename)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the cropped image
    nrrd.write(new_path, cropped_data)

def calculate_final_size(original_size):
    # Calculate final size using the provided formula
    return int(np.floor(original_size * (1 - (np.sqrt(2) - 1) * 0.5 * np.sqrt(2))))

def process_images(data_dir, output_dir):
    cta_files = _list_nrrd_files_recursively(os.path.join(data_dir, 'cta'))
    annotation_files = _list_nrrd_files_recursively(os.path.join(data_dir, 'annotation'))

    if not cta_files:
        print("No CTA files found.")
        return

    # Read the dimensions of the first image to determine the original size
    first_image = read_nrrd(cta_files[0])
    original_size = first_image.shape[1]  # Assuming the images are square (HxW)
    final_size = calculate_final_size(original_size)
    print("Original size detected: ", original_size)
    print("Calculated final size for cropping: ", final_size)

    for cta_path, annotation_path in zip(cta_files, annotation_files):
        nrrd_image = read_nrrd(cta_path)
        nrrd_class = read_nrrd(annotation_path)

        print(f"Original dimensions of {os.path.basename(cta_path)}: {nrrd_image.shape}")
        print(f"Original dimensions of {os.path.basename(annotation_path)}: {nrrd_class.shape}")

        cropped_image, cropped_class = center_crop_arr([nrrd_image, nrrd_class], final_size)

        print(f"Cropped dimensions of {os.path.basename(cta_path)}: {cropped_image.shape}")
        print(f"Cropped dimensions of {os.path.basename(annotation_path)}: {cropped_class.shape}")

        # Save cropped images
        save_cropped_image(os.path.join(output_dir, 'cta'), cta_path, cropped_image)
        save_cropped_image(os.path.join(output_dir, 'annotation'), annotation_path, cropped_class)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Cropping Script")
    parser.add_argument("--data_folder", required=True, help="Path to the data folder")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder")
    
    args = parser.parse_args()
    
    process_images(args.data_folder, args.output_folder)