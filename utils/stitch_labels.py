import os
import nibabel as nib
import numpy as np
import argparse
import glob

def join_label_maps(input_directory, output_directory):
    """
    Joins sub-volumes from different annotation types into single NIfTI files and saves them.

    Parameters:
        input_directory (str): Directory containing annotation subdirectories.
        output_directory (str): Directory where joined NIfTI files will be saved.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define annotation types and corresponding filenames
    annotation_types = [
        'coronary_annotation',
        'septal_annotation',
        'dilation_annotation',
        'hypertrophy_annotation'
    ]

    # Loop over each annotation type
    for annotation in annotation_types:
        annotation_folder = os.path.join(input_directory, annotation, 'validation')
        if not os.path.exists(annotation_folder):
            print(f"Warning: {annotation_folder} does not exist.")
            continue

        # Collect all NIfTI files in the current annotation folder
        files = sorted(glob.glob(os.path.join(annotation_folder, '759.label_*.nii.gz')))

        if len(files) != 8:
            print(f"Warning: Expected 8 files in {annotation_folder}, found {len(files)}")
            continue

        # Load and concatenate volumes along the last axis
        all_volumes = [nib.load(f).get_fdata() for f in files]
        concatenated_data = np.concatenate(all_volumes, axis=-1)

        # Save the concatenated volume
        output_filename = f"759_{annotation.split('_')[0]}.label.nii.gz"
        output_path = os.path.join(output_directory, output_filename)
        
        # Use the affine and header from the first file
        example_img = nib.load(files[0])
        concatenated_img = nib.Nifti1Image(concatenated_data, affine=example_img.affine, header=example_img.header)
        nib.save(concatenated_img, output_path)
        
        print(f"Saved joined volume for {annotation} to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Join label maps into single NIfTI files")
    parser.add_argument("--input_directory", required=True, help="Input directory containing the annotation subdirectories")
    parser.add_argument("--output_directory", required=True, help="Output directory to save joined NIfTI files")
    args = parser.parse_args()

    join_label_maps(args.input_directory, args.output_directory)
