"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import nrrd
import nibabel as nib
import numpy as np

import torch as th
import torch.distributed as dist
import torchvision as tv

from guided_diffusion.image_datasets import load_data

from guided_diffusion import dist_util, logger
import glob
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        is_train=False
    )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(args.results_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    # Directory for saving history samples
    if args.history:
        history_sample_path = os.path.join(args.results_path, 'history_samples')
        os.makedirs(history_sample_path, exist_ok=True)

    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        affine_path = cond['path']

        image = ((batch + 1.0) / 2.0).cuda() #in order to save later, we dont actually use this to sample
        label = (cond['label_ori'].float()).cuda()# / 255.0).cuda()
        model_kwargs = preprocess_input(cond, num_classes=args.num_classes)
        _, d, h, w = label.size()
        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, history_list = sample_fn(
            model,
            (args.batch_size, 1, d, h, w), #this is used to create the initial noise, so 1 channel and not nclasses channels
            args.history, #saving intermediate samples
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=False #loading animation
        )
        sample = (sample + 1) / 2.0
        for sample_i in history_list:
            sample_i = (sample_i + 1) / 2.0

        gathered_samples = [th.zeros_like(sample)]
        
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        for j in range(sample.shape[0]):
            base_filename = cond['path'][j].split('/')[-1].split('.')[0]
            #base_filename = str(len(all_samples) * args.batch_size)
            if args.dataset_mode == 'nrrd':
                # Directories for saving NRRD files
                file_image_path = os.path.join(image_path, base_filename + '.nrrd')
                file_sample_path = os.path.join(sample_path, base_filename + '.nrrd')
                file_label_path = os.path.join(label_path, base_filename + '.nrrd')
            elif args.dataset_mode == 'nifti':
                # Directories for saving NIFTI files
                file_image_path = os.path.join(image_path, base_filename + '.nii.gz')
                file_sample_path = os.path.join(sample_path, base_filename + '.nii.gz')
                file_label_path = os.path.join(label_path, base_filename + '.nii.gz')

            # Ensure directories exist
            os.makedirs(os.path.dirname(file_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(file_sample_path), exist_ok=True)
            os.makedirs(os.path.dirname(file_label_path), exist_ok=True)

            # Convert tensors to numpy arrays and squeeze if necessary
            np_image = image[j].cpu().numpy().squeeze()
            np_sample = sample[j].cpu().numpy().squeeze()
            np_label = label[j].cpu().numpy().squeeze()

            # Save the numpy arrays as NRRD files
            if args.dataset_mode == 'nrdd':
                nrrd.write(file_image_path, np_image)
                nrrd.write(file_sample_path, np_sample)
                nrrd.write(file_label_path, np_label)
            elif args.dataset_mode == 'nifti':
                affine = get_affine(args.dataset_mode, affine_path[j])
                nib.save(nib.Nifti1Image(np_image, affine), file_image_path)
                nib.save(nib.Nifti1Image(np_sample, affine), file_sample_path)
                nib.save(nib.Nifti1Image(np_label, affine), file_label_path)
            elif args.dataset_mode == 'all':
                if cond['path'][j].endswith('.nrrd'):
                    nrrd.write(file_image_path, np_image)
                    nrrd.write(file_sample_path, np_sample)
                    nrrd.write(file_label_path, np_label)
                elif cond['path'][j].endswith('.nii.gz'):
                    affine = get_affine(args.dataset_mode, affine_path[j])
                    nib.save(nib.Nifti1Image(np_image, affine), file_image_path)
                    nib.save(nib.Nifti1Image(np_sample, affine), file_sample_path)
                    nib.save(nib.Nifti1Image(np_label, affine), file_label_path)
            else:
                raise ValueError(f"Invalid dataset mode: {args.dataset_mode}")
        
        if args.history:
            for j in range(args.batch_size):
                base_filename = cond['path'][j].split('/')[-1].split('.')[0]
                for i, history_sample in enumerate(history_list):
                    # Convert tensor to numpy array and squeeze if necessary
                    np_history_sample = history_sample.cpu().numpy().squeeze()

                    if args.dataset_mode == 'nrrd':
                        # Filename for each sample in history
                        history_filename = os.path.join(history_sample_path, f'{base_filename}_history_sample_{i}.nrrd')

                        # Save the numpy array as a NRRD file
                        nrrd.write(history_filename, np_history_sample)
                    elif args.dataset_mode == 'nifti':
                        # Filename for each sample in history
                        history_filename = os.path.join(history_sample_path, f'{base_filename}_history_sample_{i}.nii.gz')

                        # Save the numpy array as a NIFTI file
                        affine = get_affine(args.dataset_mode, affine_path[j])
                        nib.save(nib.Nifti1Image(np_history_sample, affine), history_filename)

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

        if len(all_samples) * args.batch_size >= args.num_samples:
            break

    logger.log("sampling complete")


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, d, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, d, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics}


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()

def get_affine(dataset_mode, image_path):
    """
    Get affine matrix of dataset by nib loading the first image of the directory.
    If the dataset_mode is nrrd return eye(4) as affine matrix.
    """
    if dataset_mode == 'nrrd':
        affine = np.eye(4)
    elif dataset_mode == 'nifti':
        img = nib.load(image_path)
        affine = img.affine
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")
    return affine


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        s=1.0,
        history=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
