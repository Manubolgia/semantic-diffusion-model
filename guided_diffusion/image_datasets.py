import os
import random

import nrrd
import nibabel as nib

import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import zoom

from guided_diffusion import logger




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
    elif dataset_mode == 'nifti_hr' and image_size == 176:
        all_files = _list_nifti_files_recursively(os.path.join(data_dir, 'cta_processed_hr176', 'training' if is_train else 'validation'))
        classes = _list_nifti_files_recursively(os.path.join(data_dir, 'annotation_processed_hr176', 'training' if is_train else 'validation'))
        instances = None        
    elif dataset_mode == 'nifti_mosaic' and image_size == 128:
        all_files = _list_nifti_files_recursively(os.path.join(data_dir, 'cta_mosaic', 'training' if is_train else 'validation'))
        classes = _list_nifti_files_recursively(os.path.join(data_dir, 'annotation_mosaic', 'training' if is_train else 'validation'))
        instances = None   
    elif dataset_mode == 'nifti_mosaic' and image_size == 64:
        all_files = _list_nifti_files_recursively(os.path.join(data_dir, 'cta_mosaic_64', 'training' if is_train else 'validation'))
        classes = _list_nifti_files_recursively(os.path.join(data_dir, 'annotation_mosaic_64', 'training' if is_train else 'validation'))
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
        return read_affine(self.local_images[0])
        
    def create_reference(self, path):
        # Read the reference image path
        parts = path.replace('\\', '/').split('/')[-1].split('_')
        reference_path = parts[0] + '.nii.gz'

        if self.resolution == 128:
            reference_path = os.path.join(os.path.dirname(path).replace('cta_mosaic', 'cta_reference'), reference_path)
        elif self.resolution == 64:
            reference_path = os.path.join(os.path.dirname(path).replace('cta_mosaic_64', 'cta_reference'), reference_path)

        arr_reference = read_nifti(reference_path).astype(np.float32)
        reference_height, reference_width, reference_depth = arr_reference.shape

        # Extract the image index from the path
        x_index, y_index, z_index = map(int, [parts[-3], parts[-2], parts[-1].split('.')[0]])
        
        if self.resolution == 128:
            subvol_dims = (128, 128, 16)
        elif self.resolution == 64:
            subvol_dims = (64, 64, 64)

        scaling_factor = (reference_height / 256, reference_width / 256, reference_depth / 256)

        # Normalize the reference volume between -1 and 1
        min_val = arr_reference.min() #-1024
        max_val = arr_reference.max() #3071

        if arr_reference.max() != arr_reference.min():
            arr_reference[arr_reference > max_val] = max_val
            arr_reference[arr_reference < min_val] = min_val
            arr_reference = (arr_reference - min_val) / (max_val - min_val)
        else:
            arr_reference = np.zeros_like(arr_reference)

        arr_reference = 2 * arr_reference - 1
        
        # Create the global positional encoding
        global_x_position = np.arange(reference_height)/reference_height
        global_y_position = np.arange(reference_width)/reference_width
        global_z_position = np.arange(reference_depth)/reference_depth

        # Create a 3D positional embedding with the same dimensions as the reference
        global_x_embedding = np.tile(global_x_position.reshape(-1, 1, 1), (1, arr_reference.shape[1], arr_reference.shape[2]))
        global_y_embedding = np.tile(global_y_position.reshape(1, -1, 1), (arr_reference.shape[0], 1, arr_reference.shape[2]))
        global_z_embedding = np.tile(global_z_position.reshape(1, 1, -1), (arr_reference.shape[0], arr_reference.shape[1], 1))

        global_x_embedding = global_x_embedding[np.newaxis, ...]
        global_y_embedding = global_y_embedding[np.newaxis, ...]
        global_z_embedding = global_z_embedding[np.newaxis, ...]

        # Create the attention map
        
        attention_map = np.zeros_like(arr_reference)

        start_x = int(x_index*subvol_dims[0]*scaling_factor[0])
        end_x = start_x + int(subvol_dims[0]*scaling_factor[0])
        start_y = int(y_index*subvol_dims[1]*scaling_factor[1])
        end_y = start_y + int(subvol_dims[1]*scaling_factor[1])
        start_z = int(z_index*subvol_dims[2]*scaling_factor[2])
        end_z = start_z + int(subvol_dims[2]*scaling_factor[2])
        
        attention_map[start_x:end_x, :, :] += 1
        attention_map[:, start_y:end_y, :] += 1
        attention_map[:, :, start_z:end_z] += 1
        #attention_map[start_x:end_x, start_y:end_y, start_z:end_z] = 1

        margin = (int(subvol_dims[0] * scaling_factor[0]), int(subvol_dims[1] * scaling_factor[1]), int(subvol_dims[2] * scaling_factor[2]))
        
        start_margin_x = max(0, start_x - margin[0])
        end_margin_x = min(reference_height, end_x + margin[0])
        start_margin_y = max(0, start_y - margin[1])
        end_margin_y = min(reference_width, end_y + margin[1])
        start_margin_z = max(0, start_z - margin[2])
        end_margin_z = min(reference_depth, end_z + margin[2])

        for i in range(start_margin_x, start_x):
            attention_map[i, :, :] += 1 - abs(i - start_x) / margin[0]
        for i in range(end_x, end_margin_x):
            attention_map[i, :, :] += 1 - abs(i - end_x) / margin[0]

        for i in range(start_margin_y, start_y):
            attention_map[:, i, :] += 1 - abs(i - start_y) / margin[1]
        for i in range(end_y, end_margin_y):
            attention_map[:, i, :] += 1 - abs(i - end_y) / margin[1]
        
        for i in range(start_margin_z, start_z):
            attention_map[:, :, i] += 1 - abs(i - start_z) / margin[2]
        for i in range(end_z, end_margin_z):
            attention_map[:, :, i] += 1 - abs(i - end_z) / margin[2]


        #normalize the attention map
        attention_map = attention_map / attention_map.max()
        attention_map = np.clip(attention_map, 0, 1)


        attention_map = attention_map[np.newaxis, ...]

        arr_reference = arr_reference[np.newaxis, ...]

        global_embedding = np.concatenate([global_x_embedding, global_y_embedding, global_z_embedding], axis=0)

        # Concatenate the reference volume and the positional encoding
        arr_reference = np.concatenate([arr_reference, attention_map, global_embedding], axis=0)

        return arr_reference

    def create_positional_embeddings(self, path):
        # Extract x, y, and z indices from the file name
        parts = path.split('/')[-1].split('_')
        x_index, y_index, z_index = map(int, [parts[-3], parts[-2], parts[-1].split('.')[0]])
        
        # Dimensions of subvolumes
        if self.resolution == 128:
            subvol_height, subvol_width, subvol_depth = (128, 128, 16)
        elif self.resolution == 64:
            subvol_height, subvol_width, subvol_depth = (64, 64, 64)

        #full volume dimensions
        full_depth, full_height, full_width = 256, 256, 256

        #calculate the starting and ending indices of the subvolume
        z_start = int(z_index * subvol_depth)
        z_end = int(z_start + subvol_depth)
        y_start = int(y_index * subvol_height)
        y_end = int(y_start + subvol_height)
        x_start = int(x_index * subvol_width)
        x_end = int(x_start + subvol_width)
        
        global_z_position = (np.arange(z_start, z_end))/full_depth
        global_y_position = (np.arange(y_start, y_end))/full_height
        global_x_position = (np.arange(x_start, x_end))/full_width

        global_z_embedding = np.tile(global_z_position.reshape(1, 1, -1), (subvol_width, subvol_height, 1))
        global_y_embedding = np.tile(global_y_position.reshape(1, -1, 1), (subvol_width, 1, subvol_depth))
        global_x_embedding = np.tile(global_x_position.reshape(-1, 1, 1), (1, subvol_height, subvol_depth))

        global_z_embedding = global_z_embedding[np.newaxis, ...]
        global_y_embedding = global_y_embedding[np.newaxis, ...]
        global_x_embedding = global_x_embedding[np.newaxis, ...]

        global_embedding = np.concatenate([global_x_embedding, global_y_embedding, global_z_embedding], axis=0)

        return global_embedding
    
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        
        path = self.local_images[idx]
        ct_image = read_nifti(path)
        out_dict = {}

        if self.local_classes is not None:
            class_path = self.local_classes[idx]
            ct_class = read_nifti(class_path)         

        #if ct_image.shape[0] != self.resolution:
        #    arr_image, arr_class = resize_arr([ct_image, ct_class], self.resolution)
        #else:
        arr_image = ct_image
        arr_class = ct_class

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()
            arr_reference = arr_reference[:, ::-1].copy()

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
        logger.log('image: ', arr_image.shape, 'label: ', arr_class.shape, 'reference: ', out_dict['reference'].shape, 'positional_encoding: ', positional_encoding.shape)
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

