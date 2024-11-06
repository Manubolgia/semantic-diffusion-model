import os
import nibabel as nib
import numpy as np
from skimage.exposure import match_histograms
import argparse
import glob

def convert_data_precision(data, target_dtype):
    """
    Convert the data to the specified target data type.
    
    Parameters:
    - data: Numpy array, the input data to be converted.
    - target_dtype: Desired data type for the output array.

    Returns:
    - Converted data array with specified precision.
    """
    return data.astype(target_dtype)

def update_metadata(src_img, target_img):
    """
    Update the metadata of a source image to match that of a target image.
    
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
    Create a mask for identifying slices that contain non-padding values.
    
    Parameters:
    - data: Numpy array, image data.
    - padding_value: int, value used to identify padding (default -1024).
    
    Returns:
    - mask: Boolean array, True for slices containing non-padding values.
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
    - scaled_syn_data: Synthetic data scaled to match the intensity range of real data.
    """
    real_lower = np.percentile(real_data[:, :, mask], lower_percentile)
    real_upper = np.percentile(real_data[:, :, mask], upper_percentile)
    syn_lower = np.percentile(syn_data[:, :, mask], lower_percentile)
    syn_upper = np.percentile(syn_data[:, :, mask], upper_percentile)
    
    # Scale synthetic data to the range of real data
    scaled_syn_data = np.copy(syn_data)
    scaled_syn_data[:, :, mask] = ((syn_data[:, :, mask] - syn_lower) / (syn_upper - syn_lower)) * (real_upper - real_lower) + real_lower
    
    # Ensure no values fall below -1024 after scaling
    scaled_syn_data[scaled_syn_data < -1024] = -1024
    
    return scaled_syn_data

def stitch_and_normalize_volumes(directory, file_pattern, output_path, level, gt_directory=None):
    """
    Stitch together sub-volumes from a directory into a single high-resolution volume
    and normalize intensities to match a ground truth volume if provided.

    Parameters:
    - directory: str, directory containing the synthetic sub-volumes.
    - file_pattern: str, pattern to match filenames (without indices and extension).
    - output_path: str, path to save the stitched and normalized volume.
    - level: float, threshold level for synthetic data intensity capping.
    - gt_directory: str, optional directory containing real sub-volumes for histogram matching.
    """
    # Collect all files matching the pattern
    files = [f for f in os.listdir(directory) if f.startswith(file_pattern) and f.endswith('.nii.gz')]
    if not files:
        raise ValueError("No files found matching the pattern in the directory")

    # Sort files by the numerical index in the filename
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Load the first file to initialize the data array
    example_img = nib.load(os.path.join(directory, files[0]))
    data_shape = list(example_img.shape)
    num_slices = sum(nib.load(os.path.join(directory, f)).shape[-1] for f in files)

    # Adjust shape to fit all slices in the final stitched volume
    data_shape[-1] = num_slices
    stitched_data = np.zeros(data_shape, dtype=example_img.get_data_dtype())
    stitched_gt_data = np.zeros(data_shape, dtype=example_img.get_data_dtype())
    stitched_data_original = np.zeros(data_shape, dtype=example_img.get_data_dtype())

    current_slice = 0
    for f in files:
        img = nib.load(os.path.join(directory, f))
        img_data = img.get_fdata()
        num_slices = img_data.shape[-1]
        gt_img_data = None

        # Normalize synthetic data to [-1, 1] range and apply thresholding
        img_data = (img_data * 2) - 1
        img_data[img_data > 0.999] = level
        img_data = (img_data + 1) / 2.0

        if gt_directory:
            gt_file = os.path.join(gt_directory, f)
            if os.path.exists(gt_file):
                gt_img = nib.load(gt_file)
                gt_img_data = gt_img.get_fdata()
            
        # Store unprocessed synthetic data
        stitched_data_original[..., current_slice:current_slice + num_slices] = img_data

        if gt_img_data is not None:
            # Create mask for non-padding slices in ground truth
            mask = mask_non_padding_slices(gt_img_data)

            # Adjust synthetic data on non-padding slices
            matched_data = np.copy(img_data)
            if mask.any():
                matched_data[:, :, mask] = scale_and_shift(img_data, gt_img_data, mask)[:, :, mask]
           
            # Copy padding regions from ground truth data
            mask_pad = ~mask
            matched_data[:, :, mask_pad] = gt_img_data[:, :, mask_pad]
        
            stitched_data[..., current_slice:current_slice + num_slices] = matched_data
            stitched_gt_data[..., current_slice:current_slice + num_slices] = gt_img_data
            current_slice += num_slices
        else:
            stitched_data[..., current_slice:current_slice + num_slices] = img_data
            current_slice += num_slices

    # Save the final stitched and normalized image
    stitched_image = nib.Nifti1Image(stitched_data, affine=example_img.affine)
    nib.save(stitched_image, output_path)

if __name__ == '__main__':
    """
    Main function for stitching and normalizing high-resolution synthetic image volumes.
    Combines multiple sub-volumes and applies intensity normalization and optional histogram 
    matching to align synthetic data to real data characteristics. Processes either multiple 
    samples or a single specified sample ID.
    """
    parser = argparse.ArgumentParser(description="Stitch and Normalize Volumes")
    parser.add_argument("--directory", required=True, help="Path to the directory containing the sub-volumes")
    parser.add_argument("--gt_directory", help="Path to the ground truth sub-volumes for histogram matching")
    parser.add_argument("--output_path", required=True, help="Path to save the stitched volume")
    parser.add_argument("--level", type=float, default=0.9, help="Level for thresholding synthetic data intensities")
    parser.add_argument("--sample_id", type=str, default=0, help="Sample ID for stitching (use '0' to process multiple samples)")
    args = parser.parse_args()

    print('Starting stitching and normalization...')
    # Process multiple samples if sample_id is 0
    if args.sample_id == '0':
        for template in range(751, 801):
            file_pattern = str(template)
            sample_directories = glob.glob(os.path.join(args.directory, f'samples/{template}_sample*'))
            for sample_directory in sample_directories:
                output_path = os.path.join(args.output_path, f"{template}{sample_directory.split('_sample')[-1]}.img.nii.gz")
                stitch_and_normalize_volumes(sample_directory, file_pattern, output_path, args.level, args.gt_directory)
    # Process a specific sample ID
    else:
        print(args.sample_id)
        file_pattern = str(args.sample_id)
        sample_directories = glob.glob(os.path.join(args.directory, f'samples/{args.sample_id}_sample*'))
        for sample_directory in sample_directories:
            output_path = os.path.join(args.output_path, f"{args.sample_id}{sample_directory.split('_sample')[-1]}.img.nii.gz")
            print([sample_directory, file_pattern, output_path])
            stitch_and_normalize_volumes(sample_directory, file_pattern, output_path, args.level, args.gt_directory)
