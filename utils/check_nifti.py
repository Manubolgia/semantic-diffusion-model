import os
import nibabel as nib

def analyze_nifti_files(base_path):
    max_intensities = []
    total_max_intensity = 0
    count = 0

    # Target subfolders inside the base path
    for subfolder in ['training', 'validation']:
        folder_path = os.path.join(base_path, subfolder)
        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for f in files:
                    if f.endswith('.nii') or f.endswith('.nii.gz'):
                        file_path = os.path.join(root, f)
                        img = nib.load(file_path)
                        img_data = img.get_fdata()
                        max_intensity = img_data.max()
                        max_intensities.append(max_intensity)
                        total_max_intensity += max_intensity
                        count += 1
        else:
            print(f"Warning: The folder {folder_path} does not exist.")

    if count > 0:
        max_overall_intensity = max(max_intensities)
        min_max_intensity = min(max_intensities)
        avg_max_intensity = total_max_intensity / count
        print(f'Max intensity: {max_overall_intensity}')
        print(f'Min max intensity: {min_max_intensity}')
        print(f'Average max intensity: {avg_max_intensity}')
    else:
        print("No NIfTI files found.")

# Example usage:
base_path = "E:/BMC/Thesis/data/kaggle_sample_hr/cta"
analyze_nifti_files(base_path)