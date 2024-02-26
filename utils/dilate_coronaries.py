import os
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

def _list_nifti_files_recursively(data_dir):
    results = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                results.append(os.path.join(root, file))
    return results

def dilate_coronaries(annotation_path, dilation_iterations=5):
    label_map_nib = nib.load(annotation_path)
    label_map_data = label_map_nib.get_fdata()
    
    # Identifying coronary arteries (label = 8)
    coronary_mask = label_map_data == 8
    dilated_coronary_mask = binary_dilation(coronary_mask, iterations=dilation_iterations)
    
    # Replace original coronary labels with dilated ones and convert to 8-bit integer
    enhanced_label_map_data = np.where(dilated_coronary_mask, 8, label_map_data)
    enhanced_label_map_data = enhanced_label_map_data.astype(np.uint8)  # Convert to 8-bit integer

    return nib.Nifti1Image(enhanced_label_map_data, label_map_nib.affine)

def process_annotations(base_dir, output_dir_name='annotation_dilated', dilation_iterations=5):
    annotation_dir = os.path.join(base_dir, 'annotation')
    output_dir = os.path.join(base_dir, output_dir_name)
    
    for set_type in ['training', 'validation']:
        annotation_paths = _list_nifti_files_recursively(os.path.join(annotation_dir, set_type))
        
        for annotation_path in annotation_paths:
            enhanced_label_map_nib = dilate_coronaries(annotation_path, dilation_iterations)
            
            # Construct the output path
            relative_path = os.path.relpath(annotation_path, start=annotation_dir)
            save_path = os.path.join(output_dir, relative_path)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            nib.save(enhanced_label_map_nib, save_path)
            print(f"Processed and saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dilate Coronary Arteries in Annotation Label Maps")
    parser.add_argument("--data_folder", required=True, help="Path to the dataset folder containing 'annotation' directory")
    parser.add_argument("--dilation_iterations", type=int, default=5, help="Number of dilation iterations to apply")

    args = parser.parse_args()

    process_annotations(args.data_folder, dilation_iterations=args.dilation_iterations)
