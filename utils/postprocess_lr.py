import nibabel as nib
from skimage.exposure import match_histograms
import argparse
import os
import numpy as np
import glob

def read_nifti(file_path):
    """
    Read a NIFTI file from the specified path.
    
    Parameters:
    - file_path: str, path to the NIFTI file.

    Returns:
    - img: NIFTI image object.
    - data: Numpy array with image data.
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    return img, data

def update_metadata(src_img, target_img):
    """
    Update the metadata of a source image to match a target image.
    
    Parameters:
    - src_img: Source NIFTI image object (synthetic).
    - target_img: Target NIFTI image object (real).

    Returns:
    - Updated source image with matched metadata.
    """
    src_img.header['datatype'] = target_img.header['datatype']
    src_img.header['bitpix'] = target_img.header['bitpix']
    src_img.update_header()
    return src_img

def mask_non_padding_slices(data, padding_value=-1024):
    """
    Create a mask for slices that contain non-padding values.
    
    Parameters:
    - data: Numpy array, image data.
    - padding_value: int, value used to identify padding (default -1024).
    
    Returns:
    - mask: Boolean array, True for slices with non-padding values.
    """
    mask = np.any(data != padding_value, axis=(0, 1))
    return mask

def scale_and_shift(syn_data, real_data, mask, lower_percentile=1, upper_percentile=99):
    """
    Scale synthetic data to match the intensity range of real data.

    Parameters:
    - syn_data: Synthetic image data.
    - real_data: Real image data.
    - mask: Boolean array, True for slices to be scaled.
    - lower_percentile: int, lower percentile for intensity range (default 1).
    - upper_percentile: int, upper percentile for intensity range (default 99).

    Returns:
    - scaled_syn_data: Synthetic data scaled to the intensity range of real data.
    """
    real_lower = np.percentile(real_data[:, :, mask], lower_percentile)
    real_upper = np.percentile(real_data[:, :, mask], upper_percentile)
    syn_lower = np.percentile(syn_data[:, :, mask], lower_percentile)
    syn_upper = np.percentile(syn_data[:, :, mask], upper_percentile)
    
    # Scale synthetic data to the range of real data
    scaled_syn_data = np.copy(syn_data)
    scaled_syn_data[:, :, mask] = ((syn_data[:, :, mask] - syn_lower) / (syn_upper - syn_lower)) * (real_upper - real_lower) + real_lower
    
    # Ensure no values fall below -1024
    scaled_syn_data[scaled_syn_data < -1024] = -1024
    
    return scaled_syn_data

def postprocessing(sample_directory, file_pattern, output_path, args):
    """
    Perform postprocessing to adjust synthetic images to resemble real ones.

    Parameters:
    - sample_directory: Directory containing synthetic samples.
    - file_pattern: Pattern for matching synthetic files.
    - output_path: Output path for saving processed images.
    - args: Command-line arguments with additional parameters.
    """
    syn_file = [f for f in os.listdir(sample_directory) if f.startswith(file_pattern) and f.endswith('.nii.gz')][0]
    if not syn_file:
        print(f'No synthetic data found for {file_pattern}')
        return
    
    syn_img, syn_data = read_nifti(os.path.join(sample_directory, syn_file))
    real_img, real_data = read_nifti(os.path.join(args.gt_directory, syn_file))

    # Adjust synthetic data to fall within range [-1, 1] and thresholding
    syn_data = (syn_data * 2) - 1
    syn_data[syn_data > 0.999] = args.level
    syn_data = (syn_data + 1) / 2.0

    # Mask for non-padding slices
    mask = mask_non_padding_slices(real_data)

    # Scale synthetic data to match real data range
    matched_data = np.copy(syn_data)
    matched_data[:, :, mask] = scale_and_shift(syn_data, real_data, mask)[:, :, mask]

    # Match histograms in padding regions
    mask_pad = ~mask
    matched_data[:, :, mask_pad] = real_data[:, :, mask_pad]

    # Update metadata of synthetic image
    updated_syn_img = update_metadata(syn_img, real_img)

    # Save the processed image with updated metadata and adjusted intensities
    updated_syn_img = nib.Nifti1Image(matched_data, syn_img.affine, syn_img.header)
    nib.save(updated_syn_img, output_path)
    print(f'Updated image saved to {output_path}')

if __name__ == '__main__':
    """
    Main function to process synthetic image samples. It applies postprocessing 
    to match synthetic images to real image characteristics using intensity scaling, 
    histogram matching, and metadata alignment. Processes either multiple templates 
    or a single sample ID based on user input.
    """
    parser = argparse.ArgumentParser(description="Image Processing Script")
    parser.add_argument("--directory", required=True, help="Path to the directory containing the samples")
    parser.add_argument("--gt_directory", required=True, help="Path to the real data folder")
    parser.add_argument("--output_path", required=True, help="Path to the output folder")
    parser.add_argument("--level", type=float, default=0.9, help="Threshold level to cap synthetic data intensities")
    parser.add_argument("--sample_id", type=str, default=0, help="Specific sample ID for processing. Use '0' to process multiple samples.")
    args = parser.parse_args()

    # Process multiple samples if sample_id is 0
    if args.sample_id == '0':
        for template in range(751, 801):
            file_pattern = str(template)
            sample_directories = glob.glob(os.path.join(args.directory, f'samples/{template}_sample*'))
            for sample_directory in sample_directories:
                output_path = os.path.join(args.output_path, f"{template}{sample_directory.split('_sample')[-1]}.img.nii.gz")
                postprocessing(sample_directory, file_pattern, output_path, args)
    
    # Process specific sample ID
    else:
        file_pattern = str(args.sample_id)
        sample_directories = glob.glob(os.path.join(args.directory, f'samples/{args.sample_id}_sample*'))
        for sample_directory in sample_directories:
            output_path = os.path.join(args.output_path, f"{args.sample_id}{sample_directory.split('_sample')[-1]}.img.nii.gz")
            postprocessing(sample_directory, file_pattern, output_path, args)
