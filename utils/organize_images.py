import os
import shutil

# Define the source directory
source_dir = '/c:/Users/Manuel/Documents/GitHub/pmsd/utils/organize_images.py'

# Define the destination directories
cta_dir = '/c:/Users/Manuel/Documents/GitHub/pmsd/utils/cta'
annotations_dir = '/c:/Users/Manuel/Documents/GitHub/pmsd/utils/annotations_coronaries'

# Iterate over the numbers from 1 to 800
for number in range(1, 801):
    # Define the image file paths
    img_path = os.path.join(source_dir, f'{number}.img.nii.gz')
    label_path = os.path.join(source_dir, f'{number}.label.nii.gz')

    # Move the image file to the cta directory
    shutil.move(img_path, cta_dir)

    # Move the label file to the annotations_coronaries directory
    shutil.move(label_path, annotations_dir)
