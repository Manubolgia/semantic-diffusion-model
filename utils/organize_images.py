import os
import shutil
from random import shuffle

# Define the paths for the main folders
main_folders = ['cta', 'annotation']

# Number of validation images
num_validation = 50

# Create training and validation folders inside each main folder
for folder in main_folders:
    for sub_folder in ['training', 'validation']:
        os.makedirs(os.path.join(folder, sub_folder), exist_ok=True)

    # List all files in the main folder
    all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    shuffle(all_files)  # Shuffle the files to ensure randomness

    # Split files into training and validation
    validation_files = all_files[:num_validation]
    training_files = all_files[num_validation:]

    # Move files to their respective folders
    for file in validation_files:
        shutil.move(os.path.join(folder, file), os.path.join(folder, 'validation', file))

    for file in training_files:
        shutil.move(os.path.join(folder, file), os.path.join(folder, 'training', file))

print("Files have been organized into training and validation folders.")
