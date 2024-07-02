import os
import nibabel as nib
import numpy as np
from skimage.exposure import match_histograms

def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    if source.dtype.kind == 'u':
        src_lookup = source.reshape(-1)
        src_counts = np.bincount(src_lookup)
        tmpl_counts = np.bincount(template.reshape(-1))

        # omit values where the count was 0
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        src_values, src_lookup, src_counts = np.unique(
            source.reshape(-1), return_inverse=True, return_counts=True
        )
        tmpl_values, tmpl_counts = np.unique(template.reshape(-1), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size
    
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)


    return interp_a_values[src_lookup].reshape(source.shape)


def match_histograms(image, reference, *, channel_axis=None):
    """Adjust an image so that its cumulative histogram matches that of another.

    The adjustment is applied separately for each channel.

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    matched : ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.

    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    if image.ndim != reference.ndim:
        raise ValueError(
            'Image and reference must have the same number of channels.'
        )

    if channel_axis is not None:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError(
                'Number of channels in the input image and reference image must match!'
            )

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(
                image[..., channel], reference[..., channel]
            )
            matched[..., channel] = matched_channel
    else:
        # _match_cumulative_cdf will always return float64 due to np.interp
        matched = _match_cumulative_cdf(image, reference)

    if matched.dtype.kind == 'f':
        # output a float32 result when the input is float16 or float32
        out_dtype = np.float32 if image.dtype in [np.float16, np.float32] else matched.dtype
        matched = matched.astype(out_dtype, copy=False)
    return matched

def match_histograms_intensity(source, reference):
    """
    Match the histogram of the source image to that of the reference image.

    Parameters:
        source (numpy.ndarray): The source image.
        reference (numpy.ndarray): The reference image.

    Returns:
        numpy.ndarray: The source image with its histogram matched to the reference image.
    """
    return match_histograms(source, reference)


def stitch_and_normalize_volumes(directory, file_pattern, output_path, gt_directory=None):
    """
    Stitches together sub-volumes from a directory into a single Nifti image and normalizes intensities.

    Parameters:
        directory (str): Directory containing the sub-volumes.
        file_pattern (str): Pattern to match filenames (without indices and extension).
        output_path (str): Path to save the stitched volume.
        normalized_output_path (str): Path to save the normalized stitched volume.
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
        img_data[img_data>0.999] = 0.9 #0.99
        
        if gt_directory:
            gt_file = os.path.join(gt_directory, f)
            if os.path.exists(gt_file):
                gt_img = nib.load(gt_file)
                gt_img_data = gt_img.get_fdata()

        stitched_data_original[..., current_slice:current_slice + num_slices] = img_data

        if gt_img_data is not None:
            img_data = match_histograms_intensity(img_data, gt_img_data)
            
            

        stitched_data[..., current_slice:current_slice + num_slices] = img_data
        stitched_gt_data[..., current_slice:current_slice + num_slices] = gt_img_data
        current_slice += num_slices

    #stitched_data = match_histograms_intensity(stitched_data, stitched_gt_data)
    stitched_data = (stitched_data + 1) / 2.0
    # Create a new Nifti image for the original data and save it
    stitched_image = nib.Nifti1Image(stitched_data, affine=example_img.affine)
    nib.save(stitched_image, output_path)

directory = ''
gt_base_directory = ''

templates = ['751']  # List of templates to process

for template in templates:
    file_pattern = template  # Adjust this to match your actual filenames
    output_path = f'.../{template}_full.nii.gz'
    gt_directory = None#os.path.join(gt_base_directory, f'cta_{template}_norm')  # Path to ground truth subvolumes

    stitch_and_normalize_volumes(directory, file_pattern, output_path, gt_directory)
