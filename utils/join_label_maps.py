import os
import numpy as np
import nibabel as nib
import nrrd

def combine_and_label_masks(input_dir, output_dir, dataset_mode):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    for number in range(1, 801):
        # File paths for the two masks
        annotation_file = os.path.join(input_dir, 'annotation', f'{number}.label.nii.gz')
        coronary_file = os.path.join(input_dir, 'annotation_coronaries', f'{number}.label.nii.gz')

        # Load the annotation mask
        if os.path.exists(annotation_file):
            annotation_nifti = nib.load(annotation_file)
            annotation_data = annotation_nifti.get_fdata()
            affine_1 = annotation_nifti.affine
        else:
            print(f"Annotation file {number}.label.nii.gz not found.")
            continue

        # Load the coronary mask
        if os.path.exists(coronary_file):
            coronary_nifti = nib.load(coronary_file)
            affine_2 = coronary_nifti.affine
            coronary_data = coronary_nifti.get_fdata()
        else:
            print(f"Coronary file {number}.label.nii.gz not found.")
            continue

        # Combine masks
        combined_mask = annotation_data.copy()
        combined_mask[coronary_data > 0] = 8  # Assign 8 to regions in coronary mask

        # Save combined mask based on dataset_mode
        
        output_file = os.path.join(output_dir, f'{number}.combined.nii.gz' if dataset_mode == 'nifti' else f'{number}.combined.nrrd')
        if dataset_mode == 'nifti':
            if affine_1 == affine_2:
                combined_nifti = nib.Nifti1Image(combined_mask.astype(np.int16), affine=affine_1)
                nib.save(combined_nifti, output_file)
            else:
                raise ValueError("Affine matrices are not equal.")
        elif dataset_mode == 'nrrd':
            nrrd.write(output_file, combined_mask.astype(np.int16))
        print(f"Saved combined mask to {output_file}")

# Define input and output directories
input_directory = 'C://Users//Manuel//Documents//GitHub//pmsd//data//asoca_full//Normal'
output_directory = 'C://Users//Manuel//Documents//GitHub//pmsd//data//asoca_full//Combined'

# Choose dataset mode: 'nrrd' or 'nifti'
dataset_mode = 'nifti'

combine_and_label_masks(input_directory, output_directory, dataset_mode)
