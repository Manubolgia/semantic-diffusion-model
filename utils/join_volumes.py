import os
import nibabel as nib
import numpy as np
import argparse

def stitch_volumes(input_directory, output_directory, start_image_number, end_image_number):
    """
    Stitches together sub-volumes from the input directory into a single Nifti image.

    Parameters:
        input_directory (str): Directory containing the sub-volumes.
        output_directory (str): Directory to save the stitched volumes.
        start_image_number (int): Starting image number for stitching.
        end_image_number (int): Ending image number for stitching.
    """
    for image_number in range(start_image_number, end_image_number + 1):
        file_pattern = f"{image_number}.img_"
        
        # Collect all files that match the pattern
        files = [f for f in os.listdir(input_directory) if f.startswith(file_pattern) and f.endswith('.nii.gz')]
        if not files:
            print(f"No files found for image number {image_number}")
            continue
        
        # Sort files by the numerical index in the filename
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Load the first file to initialize the data array
        example_img = nib.load(os.path.join(input_directory, files[0]))
        data_shape = list(example_img.shape)
        num_slices = sum(nib.load(os.path.join(input_directory, f)).shape[-1] for f in files)
        
        # Adjust the shape to accommodate all slices
        data_shape[-1] = num_slices
        stitched_data = np.zeros(data_shape, dtype=example_img.get_data_dtype())
        
        current_slice = 0
        for f in files:
            img = nib.load(os.path.join(input_directory, f))
            img_data = img.get_fdata()
            num_slices = img_data.shape[-1]
            
            stitched_data[..., current_slice:current_slice + num_slices] = img_data
            current_slice += num_slices

        # Create a new Nifti image for the stitched data and save it
        stitched_image = nib.Nifti1Image(stitched_data, affine=example_img.affine)
        output_file = os.path.join(output_directory, f"{image_number}.nii.gz")
        nib.save(stitched_image, output_file)
        print(f"Stitched image saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stitch Volumes")
    parser.add_argument("--input_directory", required=True, help="Directory containing the sub-volumes")
    parser.add_argument("--output_directory", required=True, help="Directory to save the stitched volumes")
    parser.add_argument("--start_image_number", type=int, required=True, help="Starting image number for stitching")
    parser.add_argument("--end_image_number", type=int, required=True, help="Ending image number for stitching")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_directory, exist_ok=True)

    print('Starting stitching...')
    stitch_volumes(args.input_directory, args.output_directory, args.start_image_number, args.end_image_number)
