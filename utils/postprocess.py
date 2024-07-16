import os
import argparse
import torchio as tio
import nibabel as nib
import numpy as np
from skimage.exposure import match_histograms

def normalize_image(image_data, min_val, max_val):
    # Clip the values to be within the min and max range
    image_data = np.clip(image_data, min_val, max_val)
    
    # Normalize the data to range [0, 1]
    if max_val != min_val:
        image_data = (image_data - min_val) / (max_val - min_val)
    else:
        image_data = np.zeros_like(image_data)
    
    # Scale to range [-1, 1]
    #normalized_data = 2 * image_data - 1
    normalized_data = image_data
    return normalized_data.astype(np.float32)

def process_image(image_path, output_path, target_size, new_spacing, lr=False, reference_path=None):
    if not lr:
        # Initialize the subject with the image
        subject = tio.Subject(image=tio.ScalarImage(image_path))
    
        # Apply resampling to the desired spacing
        resampled_subject = tio.Resample(new_spacing)(subject)
    
        # Apply cropping/padding
        image_crop_or_pad = tio.CropOrPad(target_size, padding_mode=-1024)  # image padding value
        processed_subject = image_crop_or_pad(resampled_subject)
    
        # Normalize the image data and convert to float32
        min_val = processed_subject.image.data.min().item()
        max_val = processed_subject.image.data.max().item()
        processed_data_float32 = normalize_image(processed_subject.image.data.numpy(), min_val, max_val)
    
        # Create a new NIfTI image
        new_img = nib.Nifti1Image(processed_data_float32.squeeze(), affine=processed_subject.image.affine)
    
        # Save the new NIfTI image
        nib.save(new_img, output_path)
    
        print(f"Processed and saved: {output_path}")
    else:
        img = nib.load(os.path.join(image_path))
        img_data = img.get_fdata()
        example_img = nib.load(os.path.join(image_path))

        # Match histogram to reference
        if reference_path is not None:
            gt_img = nib.load(reference_path)
            gt_img_data = gt_img.get_fdata()
            img_data = match_histograms(img_data, gt_img_data)

        new_img = nib.Nifti1Image(img_data, affine=example_img.affine)
        nib.save(new_img, output_path)

def process_folder(input_folder, output_folder, target_size, new_spacing, lr=False, reference_path=None):
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_image(input_path, output_path, target_size, new_spacing, lr, reference_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample and Crop NIfTI Images in a Folder")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing NIfTI files")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder to save processed NIfTI files")
    parser.add_argument("--crop_dims", type=int, default=64, help="Dimensions to crop or pad to (default: 64)")
    parser.add_argument("--lr", type=bool, default=False, help="Whether to apply LR postprocessing (default: False)")
    parser.add_argument("--reference_path", type=str, default=None, help="Reference image path for histogram matching")

    args = parser.parse_args()
    
    new_spacing = (5, 5, 5)
    target_size = (args.crop_dims, args.crop_dims, args.crop_dims)

    os.makedirs(args.output_folder, exist_ok=True)
    process_folder(args.input_folder, args.output_folder, target_size, new_spacing, args.lr, args.reference_path)
