import os
import math
import random

import nrrd
import nibabel as nib

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
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
        all_files = _list_nrrd_files_recursively(os.path.join(data_dir, 'cta', 'training' if is_train else 'validation'))
        classes = _list_nrrd_files_recursively(os.path.join(data_dir, 'annotation', 'training' if is_train else 'validation'))
        instances = None
    elif dataset_mode == 'nifti':
        all_files = _list_nifti_files_recursively(os.path.join(data_dir, 'cta', 'training' if is_train else 'validation'))
        classes = _list_nifti_files_recursively(os.path.join(data_dir, 'annotation', 'training' if is_train else 'validation'))
        instances = None
    elif dataset_mode == 'all':
        all_files = _list_all_files_recursively(os.path.join(data_dir, 'cta', 'training' if is_train else 'validation'))
        classes = _list_all_files_recursively(os.path.join(data_dir, 'annotation', 'training' if is_train else 'validation'))
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
        random_crop=random_crop,
        random_flip=random_flip,
        is_train=is_train
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
        self.random_crop = random_crop
        self.random_flip = random_flip

    def get_affine(self):
        if self.dataset_mode == 'nrrd':
            return np.eye(4)
        elif self.dataset_mode == 'nifti':
            return read_affine(self.local_images[0])
        elif self.dataset_mode == 'all':
            if self.local_images[0].endswith('.nrrd'):
                return np.eye(4)
            elif self.local_images[0].endswith('.nii.gz'):
                return read_affine(self.local_images[0])
            else:
                raise NotImplementedError('{} not implemented'.format(self.local_images[0]))
    
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        
        path = self.local_images[idx]
        if self.dataset_mode == 'nrrd':
            ct_image = read_nrrd(path) 
        elif self.dataset_mode == 'nifti':
            ct_image = read_nifti(path)
        elif self.dataset_mode == 'all':
            if path.endswith('.nrrd'):
                ct_image = read_nrrd(path)
            elif path.endswith('.nii.gz'):
                ct_image = read_nifti(path)
            else:
                raise NotImplementedError('{} not implemented'.format(path))
        else:
            raise NotImplementedError('{} not implemented'.format(self.dataset_mode))

        out_dict = {}

        if self.local_classes is not None:
            class_path = self.local_classes[idx]
            if self.dataset_mode == 'nrrd':
                ct_class = read_nrrd(class_path)
            elif self.dataset_mode == 'nifti':
                ct_class = read_nifti(class_path)
            elif self.dataset_mode == 'all':
                if class_path.endswith('.nrrd'):
                    ct_class = read_nrrd(class_path)
                elif class_path.endswith('.nii.gz'):
                    ct_class = read_nifti(class_path)
                else:
                    raise NotImplementedError('{} not implemented'.format(class_path))
            else:
                raise NotImplementedError('{} not implemented'.format(self.dataset_mode))



        if self.is_train:
            arr_image, arr_class = resize_arr([ct_image, ct_class], self.resolution)
            #if self.random_crop:
            #    arr_image, arr_class = random_crop_arr([nrrd_image, nrrd_class], self.resolution)
            #else:
            #    arr_image, arr_class = center_crop_arr([nrrd_image, nrrd_class], self.resolution)
        else:
            arr_image, arr_class = resize_arr([ct_image, ct_class], self.resolution)
            #arr_image, arr_class = random_crop_arr([nrrd_image, nrrd_class], self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()


        arr_image = np.expand_dims(arr_image, axis=0).astype(np.float32)
        #arr_image = arr_image.astype(np.float32)
        
        min_val = arr_image.min()
        max_val = arr_image.max()

        # Normalize to [0, 1]
        arr_image = (arr_image - min_val) / (max_val - min_val)

        # Scale to [-1, 1]
        arr_image = 2 * arr_image - 1

        out_dict['path'] = path
        out_dict['label_ori'] = arr_class.copy()
        out_dict['label'] = arr_class[None, ]


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

    np_image, np_class = np_list

    def resize_image(image, target_size, resample_method):

        new_size = (target_size, target_size, target_size)

        # Calculate zoom factors for each dimension
        zoom_factors = (new_size[0] / image.shape[0], new_size[1] / image.shape[1], new_size[2] / image.shape[2])

        # Resize using zoom
        return zoom(image, zoom_factors, order=resample_method)

    arr_image = resize_image(np_image, image_size, 3)
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

