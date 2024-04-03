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


def process_images(cta_path, annotation_path, target_size, new_spacing, crop_dims, hr=False, depth=10):
    # Initialize the subject with image and label maps
    subject = tio.Subject({
        "image": tio.ScalarImage(cta_path),
        "label": tio.LabelMap(annotation_path)
    })
    
    #min_intensity_value = subject['image'][tio.DATA].min().item()
    min_intensity_value = -1024
    
    # Apply resampling and cropping/padding
    resampled_subject = tio.Resample(new_spacing)(subject)
    crop_or_pad = tio.CropOrPad((target_size, target_size, target_size), padding_mode=min_intensity_value)
    processed_subject = crop_or_pad(resampled_subject)

    # Resize for non-HR scenario
    if not hr:
        resized_subject = tio.Resize((crop_dims, crop_dims, crop_dims))(processed_subject)
        
        for key in ['image', 'label']:
            original_path = getattr(subject, key).path
            save_path = str(original_path).replace('cta', 'cta_processed').replace('annotation_dilated', 'annotation_processed')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            nib.save(nib.Nifti1Image(resized_subject[key].data.numpy().squeeze(), affine=resized_subject[key].affine), save_path)
    else:
        # Process for HR scenario with slices of depth D
        for key in ['image', 'label']:
            original_path = getattr(subject, key).path
            original_file_name = os.path.splitext(os.path.basename(original_path))[0].replace('.img.nii', '.img').replace('.label.nii', '.label')
            save_path = str(original_path).replace('cta', 'cta_processed_hr').replace('annotation_dilated', 'annotation_processed_hr')
            base_dir = os.path.dirname(save_path)
            
            # Calculate the number of sub-volumes
            num_slices = processed_subject[key].data.shape[-1]
            for start_slice in range(0, num_slices, depth):
                end_slice = min(start_slice + depth, num_slices)
                
                # Extract the sub-volume
                sub_volume_data = processed_subject[key].data[..., start_slice:end_slice]
                
                # Ensure the directory exists
                os.makedirs(base_dir, exist_ok=True)
                
                # Format the save path with sub-volume index
                sub_volume_save_path = f"{base_dir}/{original_file_name}_{start_slice//depth:04d}.nii.gz"
                
                # Save the sub-volume
                nib.save(nib.Nifti1Image(sub_volume_data.numpy().squeeze(), affine=processed_subject[key].affine), sub_volume_save_path)

                
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Script")

    parser.add_argument("--data_folder", required=True, help="Path to the preprocessed data folder")
    parser.add_argument("--crop_dims", type=int, default=64, help="Dimensions to crop or pad to (default: 64)")

    parser.add_argument("--hr", type=bool, default=False, help="Whether to process HR images (default: False)")

    args = parser.parse_args()

    new_spacing = (1.0, 1.0, 1.0)
    categories = ['cta', 'annotation_dilated']
    sets = ['training', 'validation']

    #all_images = []
    #for set_type in sets:
    #        all_images.extend(_list_nifti_files_recursively(os.path.join(args.data_folder, 'cta', set_type)))

    #target_size = calculate_target_size(all_images, new_spacing)
    #target_size = 201
    target_size = 200

    for set_type in sets:
        image_paths = _list_nifti_files_recursively(os.path.join(args.data_folder, 'cta', set_type))
        label_paths = _list_nifti_files_recursively(os.path.join(args.data_folder, 'annotation_dilated', set_type))

        # Order the paths
        image_paths.sort()
        label_paths.sort()
        
        for image_path, label_path in zip(image_paths, label_paths):
            process_images(image_path, label_path, target_size, new_spacing, args.crop_dims, args.hr)
            print(f"Processed: {image_path}, {label_path}")