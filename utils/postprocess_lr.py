import nibabel as nib
from skimage.exposure import match_histograms
import argparse
import os

def list_nifti_files(data_dir):
    """
    List all NIFTI files recursively in a directory.
    """
    results = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                results.append(os.path.join(root, file))
    return results

def read_nifti(file_path):
    """
    Read NIFTI file and return the image object and its numpy data.
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    return img, data

def update_metadata(src_img, target_img):
    """
    Update the metadata of src_img to match target_img.
    """
    src_img.header['datatype'] = target_img.header['datatype']
    src_img.header['bitpix'] = target_img.header['bitpix']
    src_img.update_header()
    return src_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Processing Script")
    parser.add_argument("--real_data_folder", required=True, help="Path to the real data folder")
    parser.add_argument("--syn_data_folder", required=True, help="Path to synthetic data folder")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder")
    parser.add_argument("--level", type=float, default=0.9, help="Level for thresholding synthetic data")
    args = parser.parse_args()

    # List files
    real_files = list_nifti_files(args.real_data_folder)
    syn_files = list_nifti_files(args.syn_data_folder)

    # Process each matched file pair
    for syn_file in syn_files:
        filename = os.path.basename(syn_file)
        real_file = os.path.join(args.real_data_folder, filename)

        if os.path.exists(real_file):
            real_img, real_data = read_nifti(real_file)
            syn_img, syn_data = read_nifti(syn_file)

            # Preprocessing synthetic data
            syn_data = (syn_data * 2) - 1
            syn_data[syn_data > 0.999] = args.level
            syn_data = (syn_data + 1) / 2.0

            # Match histograms
            matched_data = match_histograms(syn_data, real_data)

            # Update metadata
            updated_syn_img = update_metadata(syn_img, real_img)

            # Save the processed image
            updated_syn_img = nib.Nifti1Image(matched_data, syn_img.affine, syn_img.header)
            output_path = os.path.join(args.output_folder, filename)
            nib.save(updated_syn_img, output_path)
            print(f'Updated image saved to {output_path}')
        else:
            print(f"Matching real file for {filename} not found.")
