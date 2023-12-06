"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import nrrd
import torch as th
import torch.distributed as dist
import torchvision as tv

from guided_diffusion.image_datasets import load_data

from guided_diffusion import dist_util, logger
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

    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        image = ((batch + 1.0) / 2.0).cuda() #in order to save later, we dont actually use this to sample
        label = (cond['label_ori'].float()).cuda()# / 255.0).cuda()
        model_kwargs = preprocess_input(cond, num_classes=args.num_classes)
        _, d, h, w = label.size()
        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, d, h, w), #this is used to create the initial noise, so 1 channel and not nclasses channels
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )
        sample = (sample + 1) / 2.0

        gathered_samples = [th.zeros_like(sample)]
        
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        for j in range(sample.shape[0]):
            #base_filename = cond['path'][j].split('/')[-1].split('.')[0]
            base_filename = str(len(all_samples) * args.batch_size)
            # Directories for saving NRRD files
            nrrd_image_path = os.path.join(image_path, base_filename + '.nrrd')
            nrrd_sample_path = os.path.join(sample_path, base_filename + '.nrrd')
            nrrd_label_path = os.path.join(label_path, base_filename + '.nrrd')

            # Ensure directories exist
            os.makedirs(os.path.dirname(nrrd_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(nrrd_sample_path), exist_ok=True)
            os.makedirs(os.path.dirname(nrrd_label_path), exist_ok=True)

            # Convert tensors to numpy arrays and squeeze if necessary
            np_image = image[j].cpu().numpy().squeeze()
            np_sample = sample[j].cpu().numpy().squeeze()
            np_label = label[j].cpu().numpy().squeeze()

            # Save the numpy arrays as NRRD files
            nrrd.write(nrrd_image_path, np_image)
            nrrd.write(nrrd_sample_path, np_sample)
            nrrd.write(nrrd_label_path, np_label)

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
        s=1.0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
