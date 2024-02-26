import os
import argparse
import torchio as tio
import nibabel as nib
import numpy as np

def _list_nifti_files_recursively(data_dir):
    results = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                results.append(os.path.join(root, file))
    return results

def calculate_target_size(files, new_spacing):
    depths = []
    for file_path in files:
        if 'annotation' in file_path:
            continue
        image = tio.ScalarImage(file_path)
        resampled_image = tio.Resample(new_spacing)(image)
        depths.append(resampled_image.shape[2])  # Depth is the third dimension

    target_size = int(np.ceil(np.percentile(depths, 90)))  # Calculate the 90th percentile
    return target_size


def process_images(cta_path, annotation_path, target_size, new_spacing, crop_dims, base_data_folder):
    
            subject = tio.Subject({
                "image": tio.ScalarImage(cta_path),
                "label": tio.LabelMap(annotation_path)
            })

            resampled_subject = tio.Resample(new_spacing)(subject)
            crop_or_pad = tio.CropOrPad((target_size, target_size, target_size))
            processed_subject = crop_or_pad(resampled_subject)

            # Resize to 64^3
            resized_subject = tio.Resize((crop_dims, crop_dims, crop_dims))(processed_subject)
            
            # Save paths
            for key in ['image', 'label']:
                original_path = getattr(subject, key).path
                # Replace the original folder name with its '_processed' counterpart in the path
                save_path = str(original_path).replace('cta', 'cta_processed').replace('annotation_dilated', 'annotation_processed')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                nib.save(nib.Nifti1Image(resized_subject[key].data.numpy().squeeze(), affine=resized_subject[key].affine), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Script")

    parser.add_argument("--data_folder", required=True, help="Path to the preprocessed data folder")
    parser.add_argument("--crop_dims", type=int, default=64, help="Dimensions to crop or pad to (default: 64)")

    args = parser.parse_args()

    new_spacing = (1.0, 1.0, 1.0)
    categories = ['cta', 'annotation_dilated']
    sets = ['training', 'validation']

    all_images = []
    for set_type in sets:
            all_images.extend(_list_nifti_files_recursively(os.path.join(args.data_folder, 'cta', set_type)))

    target_size = calculate_target_size(all_images, new_spacing)

    print(f"Target size: {target_size}")

    for set_type in sets:
        image_paths = _list_nifti_files_recursively(os.path.join(args.data_folder, 'cta', set_type))
        label_paths = _list_nifti_files_recursively(os.path.join(args.data_folder, 'annotation_dilated', set_type))
        
        for image_path, label_path in zip(image_paths, label_paths):
            process_images(image_path, label_path, target_size, new_spacing, args.crop_dims, args.data_folder)
            print(f"Processed: {image_path}, {label_path}")