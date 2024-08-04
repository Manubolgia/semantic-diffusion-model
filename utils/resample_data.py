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
        if 'cta' in file_path:
            continue
        image = tio.ScalarImage(file_path)
        resampled_image = tio.Resample(new_spacing)(image)
        depths.append(np.min(resampled_image.shape[1:]))  # Depth is the third dimension

    target_size = int(np.ceil(np.percentile(depths, 90)))  # Calculate the 90th percentile
    return target_size

def normalize_image(image_data):
    max_val = image_data.max()
    min_val = image_data.min()
    normalized_data = 2 * ((image_data - min_val) / (max_val - min_val)) - 1
    return normalized_data

def process_images(cta_path, annotation_path, target_size_hw, target_size_d, new_spacing, crop_dims=128, hr=False, ref=False, depth=16):
    # Initialize the subject with image and label maps
    subject = tio.Subject({
        "image": tio.ScalarImage(cta_path),
        "label": tio.LabelMap(annotation_path)
    })

    # Apply resampling and cropping/padding
    resampled_subject = tio.Resample(new_spacing)(subject)
    
    processed_subject = resampled_subject

    image_crop_or_pad = tio.CropOrPad(
        (target_size_hw, target_size_hw, target_size_d),
        padding_mode=-1024  # image padding value
    )
    processed_subject['image'] = image_crop_or_pad(resampled_subject['image'])

    label_crop_or_pad = tio.CropOrPad(
        (target_size_hw, target_size_hw, target_size_d),
        padding_mode=0  # label padding value
    )
    processed_subject['label'] = label_crop_or_pad(resampled_subject['label'])
    
    # Normalize the image data
    #processed_subject['image'].set_data(normalize_image(processed_subject['image'][tio.DATA]))

    # Resize for non-HR scenario
    if not hr:
        resized_subject = tio.Resize((crop_dims, crop_dims, crop_dims))(processed_subject)
        
        for key in ['image', 'label']:
            original_path = getattr(subject, key).path
            save_path = str(original_path).replace('cta', 'cta_processed').replace('annotation_dilated', 'annotation_processed')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            nib.save(nib.Nifti1Image(resized_subject[key].data.numpy().squeeze(), affine=resized_subject[key].affine), save_path)
    else:
        if ref:
            resized_subject = tio.Resize((crop_dims, crop_dims, crop_dims))(processed_subject)
        
            for key in ['image']:
                original_path = getattr(subject, key).path
                save_path = str(original_path).replace('cta', 'cta_reference')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                nib.save(nib.Nifti1Image(resized_subject[key].data.numpy().squeeze(), affine=resized_subject[key].affine), save_path)
        else:
            processed_subject = tio.Resize((crop_dims, crop_dims, crop_dims), image_interpolation='linear')(processed_subject)
            # Process for HR scenario with slices of depth D
            for key in ['image', 'label']:
                original_path = getattr(subject, key).path
                original_file_name = os.path.splitext(os.path.basename(original_path))[0].replace('.img.nii', '.img').replace('.label.nii', '.label')
                save_path = str(original_path).replace('cta', 'cta_processed_hr').replace('annotation', 'annotation_processed_hr')
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
    parser.add_argument("--ref", type=bool, default=False, help="Whether to process the reference images (default: False)")

    args = parser.parse_args()

    new_spacing = (1.0, 1.0, 1.0)
    sets = ['training', 'validation']

    target_size_hw = 160
    target_size_d = 160

    for set_type in sets:
        image_paths = _list_nifti_files_recursively(os.path.join(args.data_folder, 'cta', set_type))
        label_paths = _list_nifti_files_recursively(os.path.join(args.data_folder, 'annotation', set_type))
        #label_paths = _list_nifti_files_recursively(os.path.join(args.data_folder, 'annotation_dilated', set_type))

        image_paths.sort()
        label_paths.sort()
        
        for image_path, label_path in zip(image_paths, label_paths):
            process_images(image_path, label_path, target_size_hw, target_size_d, new_spacing, args.crop_dims, args.hr, args.ref)
            print(f"Processed: {image_path}, {label_path}")
