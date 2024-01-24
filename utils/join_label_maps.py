import os
import numpy as np
import nibabel as nib
import nrrd


def combine_and_label_masks(input_dir, output_dir, label_mapping):

    # Iterate through each subfolder in the input directory
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        if os.path.isdir(folder_path):
            combined_mask = None

            # Handle the "Normal_n" structure
            coronary_file = f'{folder_name}.nii.gz'
            coronary_path = os.path.join(folder_path, coronary_file)
            if os.path.exists(coronary_path):
                coronary_nifti = nib.load(coronary_path)
                coronary_data = coronary_nifti.get_fdata()

                combined_mask = np.zeros_like(coronary_data)
                combined_mask[coronary_data > 0] = label_mapping['Coronaries']

            # Handle other structures
            for structure, label in label_mapping.items():
                if structure == 'Coronaries':
                    continue  # Skip since it's already handled

                mask_file = f'{structure}.nii.gz'
                mask_path = os.path.join(folder_path, mask_file)

                if not os.path.exists(mask_path):
                    print(f"Mask file for {structure} not found in {folder_name}.")
                    continue

                mask_nifti = nib.load(mask_path)
                mask_data = mask_nifti.get_fdata()

                # Assign the label to the combined mask
                combined_mask[mask_data > 0] = label

            # Save combined mask as NRRD
            output_file = os.path.join(output_dir, folder_name + '.nrrd')
            nrrd.write(output_file, combined_mask.astype(np.int16))
            print(f"Saved combined mask to {output_file}")

input_directory = 'C://Users//Manuel//Documents//GitHub//pmsd//data//asoca_full//Normal//Annotations'
output_directory = 'C://Users//Manuel//Documents//GitHub//pmsd//data//asoca_full//Normal//Annotations'
label_mapping = {
    'heart_atrium_left': 3,
    'heart_atrium_right': 4,
    'heart_ventricle_left': 6,
    'heart_ventricle_right': 7,
    'heart_myocardium': 5,
    'aorta': 2,
    'pulmonary_artery': 8,
    'Coronaries': 1,  # Coronaries
    # Add more structures and their labels as needed
}

combine_and_label_masks(input_directory, output_directory, label_mapping)
