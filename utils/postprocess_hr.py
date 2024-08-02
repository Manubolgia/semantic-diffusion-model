import os
import nibabel as nib
import numpy as np
from skimage.exposure import match_histograms
import argparse
import glob

def convert_data_precision(data, target_dtype):
    """
    Convert the data to the target data type.
    """
    return data.astype(target_dtype)

def update_metadata(src_img, target_img):
    """
    Update the metadata of src_img to match target_img.
    """
    src_img.header['datatype'] = target_img.header['datatype']
    src_img.header['bitpix'] = target_img.header['bitpix']
    src_img.update_header()
    return src_img

def mask_non_padding_slices(data, padding_value=-1024):
    """
    Create a mask that identifies non-padding slices.
    """
    mask = np.any(data != padding_value, axis=(0, 1))
    return mask

def scale_and_shift(syn_data, real_data, mask, lower_percentile=1, upper_percentile=99):
    real_lower = np.percentile(real_data[:, :, mask], lower_percentile)
    real_upper = np.percentile(real_data[:, :, mask], upper_percentile)
    syn_lower = np.percentile(syn_data[:, :, mask], lower_percentile)
    syn_upper = np.percentile(syn_data[:, :, mask], upper_percentile)
    
    # Scale synthetic data to the range of real data
    scaled_syn_data = np.copy(syn_data)
    scaled_syn_data[:, :, mask] = ((syn_data[:, :, mask] - syn_lower) / (syn_upper - syn_lower)) * (real_upper - real_lower) + real_lower
    
    # Ensure no values are below -1024 after scaling
    scaled_syn_data[scaled_syn_data < -1024] = -1024
    
    return scaled_syn_data

def stitch_and_normalize_volumes(directory, file_pattern, output_path, level, gt_directory=None):
    """
    Stitches together sub-volumes from a directory into a single Nifti image and normalizes intensities.

    Parameters:
        directory (str): Directory containing the sub-volumes.
        file_pattern (str): Pattern to match filenames (without indices and extension).
        output_path (str): Path to save the stitched volume.
        gt_directory (str): Directory containing the ground truth sub-volumes for histogram matching.
    """
    # Collect all files that match the pattern
    files = [f for f in os.listdir(directory) if f.startswith(file_pattern) and f.endswith('.nii.gz')]
    if not files:
        raise ValueError("No files found matching the pattern in the directory")

    # Sort files by the numerical index in the filename
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Load the first file to initialize the data array
    example_img = nib.load(os.path.join(directory, files[0]))
    data_shape = list(example_img.shape)
    num_slices = sum(nib.load(os.path.join(directory, f)).shape[-1] for f in files)

    # Adjust the shape to accommodate all slices
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

        img_data = (img_data*2) - 1
        img_data[img_data>0.999] = level # 0.9 #0.99
        img_data = (img_data + 1) / 2.0
        
        if gt_directory:
            gt_file = os.path.join(gt_directory, f)
            if os.path.exists(gt_file):
                gt_img = nib.load(gt_file)
                gt_img_data = gt_img.get_fdata()
            

        stitched_data_original[..., current_slice:current_slice + num_slices] = img_data

        if gt_img_data is not None:
            # Create mask for non-padding slices in the ground truth data
            mask = mask_non_padding_slices(gt_img_data)

            # Histogram matching on non-padding slices
            matched_data = np.copy(img_data)
            if mask.any():
                #matched_data[:, :, mask] = match_histograms(img_data[:, :, mask], gt_img_data[:, :, mask])

                # Alternative method: scaling and shifting using percentiles
                matched_data[:, :, mask] = scale_and_shift(img_data, gt_img_data, mask)[:, :, mask]
           
            mask_pad = ~mask
            matched_data[:, :, mask_pad] = gt_img_data[:, :, mask_pad]
        
            
            stitched_data[..., current_slice:current_slice + num_slices] = matched_data
            stitched_gt_data[..., current_slice:current_slice + num_slices] = gt_img_data
            current_slice += num_slices
        
        else:
            stitched_data[..., current_slice:current_slice + num_slices] = img_data
            current_slice += num_slices


    # Create a new Nifti image for the original data and save it
    stitched_image = nib.Nifti1Image(stitched_data, affine=example_img.affine)
    nib.save(stitched_image, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stitch and Normalize Volumes")
    parser.add_argument("--directory", required=True, help="Directory containing the sub-volumes")
    parser.add_argument("--gt_directory", help="Directory containing the ground truth sub-volumes for histogram matching")
    parser.add_argument("--output_path", required=True, help="Path to save the stitched volume")
    parser.add_argument("--level", type=float, default=0.9, help="Level for thresholding synthetic data")
    parser.add_argument("--sample_id", type=str, default=0, help="Sample ID for stitching")
    args = parser.parse_args()

    if args.sample_id == '0':
        for template in range(751, 801):
            file_pattern = str(template)
            sample_directories = glob.glob(os.path.join(args.directory, f'samples/{template}_sample*'))
            for sample_directory in sample_directories:
                output_path = os.path.join(args.output_path, f"{template}_{sample_directory.split('_sample')[-1]}.img.nii.gz")
                stitch_and_normalize_volumes(sample_directory, file_pattern, output_path, args.level, args.gt_directory)
    else:
        file_pattern = str(args.sample_id)
        sample_directories = glob.glob(os.path.join(args.directory, f'samples/{args.sample_id}_sample*'))
        for sample_directory in sample_directories:
            output_path = os.path.join(args.output_path, f"{args.sample_id}_{sample_directory.split('_sample')[-1]}.img.nii.gz")
            print([sample_directory, file_pattern, output_path])
            stitch_and_normalize_volumes(sample_directory, file_pattern, output_path, args.level, args.gt_directory)