import os
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator

def process_folder_with_totalsegmentator(input_folder, output_folder, task_name):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each NIfTI file in the input folder
    for file in os.listdir(input_folder):
        if file.endswith('.nii.gz'):
            print(file)
            input_file = input_folder / file
            
            # Extract the number from the input file name
            number = file.split('.')[0]
            
            output_file = output_folder / (number + '.label.nii.gz')

            # Call TotalSegmentator for the file
            totalsegmentator(str(input_file), str(output_file), task=task_name, ml=True)

            # If additional processing of label maps is needed, it can be done here

if __name__ == '__main__':
    input_folder = 'C:/Users/Manuel/Documents/GitHub/pmsd/data/testing_seg'
    output_folder = 'C:/Users/Manuel/Documents/GitHub/pmsd/data/testing_seg/annotations'
    task_name = 'total'
    process_folder_with_totalsegmentator(input_folder, output_folder, task_name)
