import os
import math
import random

import nrrd
import nibabel as nib

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from scipy.ndimage import zoom



def load_data(
    *,
    dataset_mode,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False, #augmentation
    random_flip=False,
    is_train=True,
    reference=False,
    pos_emb=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if dataset_mode == 'nrrd':
        all_files = _list_nrrd_files_recursively(os.path.join(data_dir, 'cta_processed', 'training' if is_train else 'validation'))
        classes = _list_nrrd_files_recursively(os.path.join(data_dir, 'annotation_processed', 'training' if is_train else 'validation'))
        instances = None
    elif dataset_mode == 'nifti':
        all_files = _list_nifti_files_recursively(os.path.join(data_dir, 'cta_processed', 'training' if is_train else 'validation'))
        classes = _list_nifti_files_recursively(os.path.join(data_dir, 'annotation_processed', 'training' if is_train else 'validation'))
        instances = None
    elif dataset_mode == 'nifti_hr' and image_size != 176:
        all_files = _list_nifti_files_recursively(os.path.join(data_dir, 'cta_processed_hr', 'training' if is_train else 'validation'))
        classes = _list_nifti_files_recursively(os.path.join(data_dir, 'annotation_processed_hr', 'training' if is_train else 'validation'))
        instances = None    
    #elif dataset_mode == 'nifti_hr' and image_size != 176:
    #    all_files = _list_nifti_files_recursively(os.path.join(data_dir, 'cta_hr', 'training' if is_train else 'validation'))
    #    classes = _list_nifti_files_recursively(os.path.join(data_dir, 'annotation_hr', 'training' if is_train else 'validation'))
    #    instances = None 
    elif dataset_mode == 'nifti_hr' and image_size == 176:
        all_files = _list_nifti_files_recursively(os.path.join(data_dir, 'cta_processed_hr176', 'training' if is_train else 'validation'))
        classes = _list_nifti_files_recursively(os.path.join(data_dir, 'annotation_processed_hr176', 'training' if is_train else 'validation'))
        instances = None        
    elif dataset_mode == 'all':
        all_files = _list_all_files_recursively(os.path.join(data_dir, 'cta_processed', 'training' if is_train else 'validation'))
        classes = _list_all_files_recursively(os.path.join(data_dir, 'annotation_processed', 'training' if is_train else 'validation'))
        instances = None
    else:
        raise NotImplementedError('{} not implemented'.format(dataset_mode))

    print("Len of Dataset:", len(all_files))

    dataset = ImageDataset(
        dataset_mode,
        image_size,
        all_files,
        classes=classes,
        instances=instances,
        reference=reference,
        random_crop=random_crop,
        random_flip=random_flip,
        is_train=is_train,
        pos_emb=pos_emb
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def _list_nrrd_files_recursively(data_dir):
    """""
    List all nrrd files recursively
        
    """""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() == "nrrd":
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_nrrd_files_recursively(full_path))
    return results

def _list_nifti_files_recursively(data_dir):
    """""
    List all nifti files recursively
    
    """""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() == "gz":
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_nifti_files_recursively(full_path))
    return results

def _list_all_files_recursively(data_dir):
    """""
    List all nrrd and nifti files recursively

    """""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and (ext.lower() == "nrrd" or ext.lower() == "gz"):
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_all_files_recursively(full_path))
    return results

class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_mode,
        resolution,
        image_paths,
        classes=None,
        instances=None,
        reference=False,
        pos_emb=False,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
        is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.local_instances = None if instances is None else instances[shard:][::num_shards]
        self.local_reference = reference
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.pos_emb = pos_emb

    def get_affine(self):
        if self.dataset_mode == 'nrrd':
            return np.eye(4)
        elif self.dataset_mode == 'nifti' or self.dataset_mode == 'nifti_hr':
            return read_affine(self.local_images[0])
        else:
            raise NotImplementedError('{} not implemented'.format(self.dataset_mode))
    
    def create_reference(self, path):
        # Read the reference image path
        reference_path = path.replace('\\', '/').split('/')[-1].split('_')[0] + '.nii.gz'
        if self.resolution == 176:
            reference_path = os.path.join(os.path.dirname(path).replace('cta_processed_hr176', 'cta_reference'), reference_path)
        else:
            #reference_path = os.path.join(os.path.dirname(path).replace('cta_processed_hr', 'cta_reference_syn32'), reference_path)
            reference_path = os.path.join(os.path.dirname(path).replace('cta_processed_hr', 'cta_reference_syn'), reference_path)
            #reference_path = os.path.join(os.path.dirname(path).replace('cta_processed_hr', 'cta_reference'), reference_path)
            #reference_path = os.path.join(os.path.dirname(path).replace('cta_hr', 'cta_reference'), reference_path)

        # Load the appropriate dataset mode
        if self.dataset_mode == 'nrrd':
            arr_reference = read_nrrd(reference_path)
        elif self.dataset_mode == 'nifti' or self.dataset_mode == 'nifti_hr':
            arr_reference = read_nifti(reference_path)
        else:
            raise NotImplementedError('{} not implemented'.format(self.dataset_mode))

        arr_reference = arr_reference.astype(np.float32)

        # Extract the image index from the path
        index = int(path.split('/')[-1].split('_')[-1].split('.')[0])

        depth = 16  # Define the depth per HR patch
        # Determine the scaling factor and the number of images based on the resolution
        if self.resolution == 176:
            original_depth = 144
            lr_depth = arr_reference.shape[2]
        elif self.resolution == 128:
            original_depth = 128
            lr_depth = arr_reference.shape[2]
        else:
            raise NotImplementedError('Resolution not implemented')

        # Calculate the starting and ending indices of the slice in the reference image
        z_start = int(index*depth*lr_depth/original_depth)
        z_end = int(z_start + depth*lr_depth/original_depth)

        # Add margin before and after the slice
        if lr_depth == 64:
            margin = 8
        elif lr_depth == 32:
            margin = 4
        z_start -= margin
        z_end += margin

        # Determine how much padding is needed before and after the volume
        pad_before = max(0, -z_start)
        pad_after = max(0, z_end - lr_depth)

        # Apply padding to the reference volume if necessary
        if pad_before > 0 or pad_after > 0:
            arr_reference = np.pad(arr_reference, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant', constant_values=-1024)

        # Adjust the start and end indices after padding
        z_start += pad_before
        z_end += pad_before

        # Normalize the reference volume between -1 and 1
        min_val = arr_reference.min() #-1024
        max_val = arr_reference.max() #3071

        if arr_reference.max() != arr_reference.min():
            arr_reference[arr_reference > max_val] = max_val
            arr_reference[arr_reference < min_val] = min_val
            arr_reference = (arr_reference - min_val) / (max_val - min_val)
        else:
            arr_reference = np.zeros_like(arr_reference)

        #0 - 1 -> -1 - 1
        arr_reference = 2 * arr_reference - 1
        
        arr_reference = arr_reference[:, :, z_start:z_end]
        arr_reference = arr_reference[np.newaxis, ...]

        # Create the global positional encoding

        global_z_position= np.arange(z_start-pad_before, z_end-pad_before)/lr_depth
        global_z_position = 2 * (global_z_position - 0.5)

        # Create a 3D positional embedding with the same dimensions as the reference
        global_z_embedding = np.tile(global_z_position.reshape(1, 1, arr_reference.shape[-1]), (lr_depth, lr_depth, 1))
        global_z_embedding = global_z_embedding[np.newaxis, ...]

        # Concatenate the reference volume and the positional encoding
        arr_reference = np.concatenate([arr_reference, global_z_embedding], axis=0)

        return arr_reference

    def create_positional_embeddings(self, path):
        #get the index of the image
        index = int(path.split('/')[-1].split('_')[-1].split('.')[0])
        
        depth = 16

        #create the positional embeddings
        z_start = int(index * depth)
        z_end = int(z_start + depth)

        global_z_position = (np.arange(z_start, z_end))/self.resolution
        global_z_position = 2 * (global_z_position - 0.5)
        global_z_embedding = np.tile(global_z_position.reshape(1, 1, depth), (self.resolution, self.resolution, 1))

        global_z_embedding = global_z_embedding[np.newaxis, ...]
        
        return global_z_embedding
    
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        
        path = self.local_images[idx]
        if self.dataset_mode == 'nrrd':
            ct_image = read_nrrd(path) 
        elif self.dataset_mode == 'nifti' or self.dataset_mode == 'nifti_hr':
            ct_image = read_nifti(path)
        else:
            raise NotImplementedError('{} not implemented'.format(self.dataset_mode))

        out_dict = {}

        if self.local_classes is not None:
            class_path = self.local_classes[idx]
            if self.dataset_mode == 'nrrd':
                ct_class = read_nrrd(class_path)
            elif self.dataset_mode == 'nifti' or self.dataset_mode == 'nifti_hr':
                ct_class = read_nifti(class_path)
            else:
                raise NotImplementedError('{} not implemented'.format(self.dataset_mode))
            

        if ct_image.shape[0] != self.resolution:
            arr_image, arr_class = resize_arr([ct_image, ct_class], self.resolution)
        else:
            arr_image = ct_image
            arr_class = ct_class

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()

        arr_image = np.expand_dims(arr_image, axis=0).astype(np.float32)
        
        min_val = arr_image.min() #-1024
        max_val = arr_image.max() #3071 

        # Normalize to [0, 1]
        if max_val != min_val:
            arr_image[arr_image > max_val] = max_val
            arr_image[arr_image < min_val] = min_val
            arr_image = (arr_image - min_val) / (max_val - min_val)
        else:
            arr_image = np.zeros_like(arr_image)

        # Scale to [-1, 1]
        arr_image = 2 * arr_image - 1

        # Positional encoding
        # -------------------
        if self.pos_emb is not False:
            positional_encoding = self.create_positional_embeddings(path)
            out_dict['positional_encoding'] = positional_encoding
        # -------------------

        out_dict['path'] = path
        out_dict['label_ori'] = arr_class.copy()
        out_dict['label'] = arr_class[None, ]
        
        # Create reference
        # -------------------
        if self.local_reference is not False:
             out_dict['reference'] = self.create_reference(path)
        # -------------------

        return arr_image, out_dict 

def read_nrrd(file_path):
    """
    Read nrrd file and return numpy array
    """
    data, header = nrrd.read(file_path)
    return data

def read_nifti(file_path):
    """
    Read nifti file and return numpy array, analog to read_nrrd
    """
    data = nib.load(file_path).get_fdata()
    return data

def read_affine(file_path):
    """
    Read nifti file and return numpy array, analog to read_nrrd
    """
    affine = nib.load(file_path).affine
    return affine

def resize_arr(np_list, image_size):

    if len(np_list)>1:
        np_image, np_class = np_list
    elif len(np_list)==1:
        np_image = np_list[0]
        np_class = None

    def resize_image(image, target_size, resample_method):

        new_size = (target_size, target_size, target_size)

        # Calculate zoom factors for each dimension
        zoom_factors = (new_size[0] / image.shape[0], new_size[1] / image.shape[1], new_size[2] / image.shape[2])

        # Resize using zoom
        return zoom(image, zoom_factors, order=resample_method)

    arr_image = resize_image(np_image, image_size, 3)
    arr_class = None
    if np_class is not None:
        arr_class = resize_image(np_class, image_size, 0)

    return arr_image, arr_class



def center_crop_arr(np_list, volume_size):

    np_image, np_class = np_list

    D, H, W = np_image.shape
    crop_D, crop_H, crop_W = [volume_size, volume_size, volume_size]

    # Ensure the crop size is smaller than the image size
    crop_D = min(crop_D, D)
    crop_H = min(crop_H, H)
    crop_W = min(crop_W, W)

    start_D = (D - crop_D) // 2
    start_H = (H - crop_H) // 2
    start_W = (W - crop_W) // 2

    end_D = start_D + crop_D
    end_H = start_H + crop_H
    end_W = start_W + crop_W

    cropped_image = np_image[start_D:end_D, start_H:end_H, start_W:end_W]
    cropped_class = np_class[start_D:end_D, start_H:end_H, start_W:end_W]

    return cropped_image, cropped_class

def random_crop_arr(np_list, volume_size):
    np_image, np_class = np_list
    D, H, W = np_image.shape
    crop_D, crop_H, crop_W = [volume_size, volume_size, volume_size]

    crop_D = min(crop_D, D)
    crop_H = min(crop_H, H)
    crop_W = min(crop_W, W)

    start_D = random.randrange(D - crop_D + 1)
    start_H = random.randrange(H - crop_H + 1)
    start_W = random.randrange(W - crop_W + 1)

    end_D = start_D + crop_D
    end_H = start_H + crop_H
    end_W = start_W + crop_W

    cropped_image = np_image[start_D:end_D, start_H:end_H, start_W:end_W]
    cropped_class = np_class[start_D:end_D, start_H:end_H, start_W:end_W]

    return cropped_image, cropped_class

