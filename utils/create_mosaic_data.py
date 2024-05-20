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


def process_images(cta_path, annotation_path, target_size_hw, target_size_d, new_spacing, crop_dims=256, subvol_dims=(64, 64, 64)):
    # 1. Resample the images to the new spacing
    # 2. Crop or pad the images to the target size
    # 3. Resize the images 
    # 4. Split the images into a mosaic of sub-volumes
    
    
    # Initialize the subject with image and label maps
    subject = tio.Subject({
        "image": tio.ScalarImage(cta_path),
        "label": tio.LabelMap(annotation_path)
    })
    
    #min_intensity_value = subject['image'][tio.DATA].min().item()
    image_padding_value = -1024
    label_padding_value = 0
    
    # Apply resampling and cropping/padding
    resampled_subject = tio.Resample(new_spacing)(subject)
    
    processed_subject = resampled_subject

    image_crop_or_pad = tio.CropOrPad(
        (target_size_hw, target_size_hw, target_size_d),
        padding_mode=image_padding_value
    )
    processed_subject['image'] = image_crop_or_pad(resampled_subject['image'])

    # For the label map: Apply cropping/padding with 0 or another appropriate value
    label_crop_or_pad = tio.CropOrPad(
        (target_size_hw, target_size_hw, target_size_d),
        padding_mode=label_padding_value
    )
    processed_subject['label'] = label_crop_or_pad(resampled_subject['label'])

    #----160x160x160----#
    processed_subject = tio.Resize((crop_dims, crop_dims, crop_dims), image_interpolation='linear')(processed_subject)
    #----256x256x256----#
    

    # Create mosaic
    for key in ['image', 'label']:
        data = processed_subject[key].data
        original_path = getattr(subject, key).path
        original_file_name = os.path.splitext(os.path.basename(original_path))[0].replace('.img.nii', '.img').replace('.label.nii', '.label')
        save_path = str(original_path).replace('cta', 'cta_mosaic_64').replace('annotation', 'annotation_mosaic_64')
        base_dir = os.path.dirname(save_path)

        # Ensure the directory exists
        os.makedirs(base_dir, exist_ok=True)
            
        # Calculate the number of sub-volumes
        num_subvols_x = data.shape[-3] // subvol_dims[0]
        num_subvols_y = data.shape[-2] // subvol_dims[1]
        num_subvols_z = data.shape[-1] // subvol_dims[2]

        # Extract and save sub-volumes
        for z in range(num_subvols_z):
            for y in range(num_subvols_y):
                for x in range(num_subvols_x):
                    start_x = x * subvol_dims[0]
                    start_y = y * subvol_dims[1]
                    start_z = z * subvol_dims[2]
                    end_x = start_x + subvol_dims[0]
                    end_y = start_y + subvol_dims[1]
                    end_z = start_z + subvol_dims[2]
                    

                    sub_volume_data = data[:, start_x:end_x, start_y:end_y, start_z:end_z]

                    sub_volume_save_path = f"{base_dir}/{original_file_name}_{x:02d}_{y:02d}_{z:02d}.nii.gz"
                    
                    nib.save(nib.Nifti1Image(sub_volume_data.numpy().squeeze(), affine=processed_subject[key].affine), sub_volume_save_path)


            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Script")

    parser.add_argument("--data_folder", required=True, help="Path to the preprocessed data folder")
    args = parser.parse_args()

    new_spacing = (1.0, 1.0, 1.0)
    sets = ['training', 'validation']

    #all_images = []
    #for set_type in sets:
    #    all_images.extend(_list_nifti_files_recursively(os.path.join(args.data_folder, 'annotation_dilated', set_type)))

    #target_size = calculate_target_size(all_images, new_spacing)
    #print(f"Target size: {target_size}")
    
    target_size_hw = 160 #176
    target_size_d = 160 #144
    

    for set_type in sets:
        image_paths = _list_nifti_files_recursively(os.path.join(args.data_folder, 'cta', set_type))
        label_paths = _list_nifti_files_recursively(os.path.join(args.data_folder, 'annotation', set_type))

        # Order the paths
        image_paths.sort()
        label_paths.sort()
        
        for image_path, label_path in zip(image_paths, label_paths):
            process_images(image_path, label_path, target_size_hw, target_size_d, new_spacing, crop_dims=256)
            print(f"Processed: {image_path}, {label_path}")