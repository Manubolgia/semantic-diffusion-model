import os
import random
import shutil

# Define the base directory
base_dir = '/home/data/farid/vessel_segmentation/kaggle_dataset/'

# Define source and target directories
cta_processed_dir = os.path.join(base_dir, 'cta_processed/training')
cta_processed_finetune_dir = os.path.join(base_dir, 'cta_processed_finetune/validation')

annotation_processed_dir = os.path.join(base_dir, 'annotation_processed/training')
annotation_processed_finetune_dir = os.path.join(base_dir, 'annotation_processed_finetune/validation')

cta_processed_hr_dir = os.path.join(base_dir, 'cta_processed_hr/training')
cta_processed_hr_finetune_dir = os.path.join(base_dir, 'cta_processed_hr_finetune/training')

annotation_processed_hr_dir = os.path.join(base_dir, 'annotation_processed_hr/training')
annotation_processed_hr_finetune_dir = os.path.join(base_dir, 'annotation_processed_hr_finetune/training')

# Create target directories if they don't exist
os.makedirs(cta_processed_finetune_dir, exist_ok=True)
os.makedirs(annotation_processed_finetune_dir, exist_ok=True)
os.makedirs(cta_processed_hr_finetune_dir, exist_ok=True)
os.makedirs(annotation_processed_hr_finetune_dir, exist_ok=True)

# Get a list of all image files in the cta_processed training directory
cta_files = [f for f in os.listdir(cta_processed_dir) if f.endswith('.img.nii.gz')]

# Select 100 random files
random_files = random.sample(cta_files, 100)

# Function to move files
def move_files(file_list, src_dir, dest_dir, suffix):
    for file in file_list:
        src_file = os.path.join(src_dir, file.replace('.img.nii.gz', suffix))
        dest_file = os.path.join(dest_dir, file.replace('.img.nii.gz', suffix))
        shutil.copy(src_file, dest_file)

# Function to copy high-resolution files
def copy_hr_files(file_list, src_dir, dest_dir, suffix):
    for file in file_list:
        base_name = file.replace('.img.nii.gz', '')
        for i in range(8):  # Assuming 0000 to 0007
            src_file = os.path.join(src_dir, f"{base_name}{suffix}_{i:04d}.nii.gz")
            dest_file = os.path.join(dest_dir, f"{base_name}{suffix}_{i:04d}.nii.gz")
            if os.path.exists(src_file):
                shutil.copy(src_file, dest_file)

# Move the selected files to the finetuning directories
move_files(random_files, cta_processed_dir, cta_processed_finetune_dir, '.img.nii.gz')
move_files(random_files, annotation_processed_dir, annotation_processed_finetune_dir, '.label.nii.gz')

# Copy the corresponding high-resolution files
copy_hr_files(random_files, cta_processed_hr_dir, cta_processed_hr_finetune_dir, '.img')
copy_hr_files(random_files, annotation_processed_hr_dir, annotation_processed_hr_finetune_dir, '.label')

print("Files have been moved successfully.")
