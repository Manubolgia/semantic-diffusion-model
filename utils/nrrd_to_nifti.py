import os
import nrrd
import nibabel as nib
import numpy as np

def convert_nrrd_to_nifti(input_dir, output_dir):
    for subfolder in ['Diseased', 'Normal']:
        ct_folder = os.path.join(input_dir, subfolder, 'CTCA')
        output_subfolder = os.path.join(output_dir, subfolder, 'CTCA_NIFTI')

        if not os.path.exists(ct_folder):
            print(f"Ct folder not found in {subfolder}")
            continue

        os.makedirs(output_subfolder, exist_ok=True)

        for filename in os.listdir(ct_folder):
            if filename.endswith('.nrrd'):
                nrrd_path = os.path.join(ct_folder, filename)
                nifti_filename = filename.replace('.nrrd', '.nii.gz')

                # Read NRRD
                nrrd_image, _ = nrrd.read(nrrd_path)

                # Convert to NIfTI
                nifti_image = nib.Nifti1Image(nrrd_image, affine=np.eye(4))

                # Save NIfTI
                nib.save(nifti_image, os.path.join(output_subfolder, nifti_filename))

input_directory = 'C://Users//Manuel//Documents//GitHub//pmsd//data//asoca_full'
output_directory = 'C://Users//Manuel//Documents//GitHub//pmsd//data//asoca_full'
convert_nrrd_to_nifti(input_directory, output_directory)
