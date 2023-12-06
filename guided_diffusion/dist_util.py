"""
Helpers for distributed training.
"""

import os
import torch as th
import blobfile as bf
import io

# Adjust this depending on your system setup
GPUS_PER_NODE = 1

def setup_dist():
    """
    Setup for a single GPU environment.
    """
    if th.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    else:
        raise RuntimeError("CUDA is not available. A GPU is required for this setup.")

def dev():
    """
    Get the device to use for PyTorch operations.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:0")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch model state dict from a file.
    This version is adapted for single GPU setups without MPI.
    """
    # Open the file and read the data
    with bf.BlobFile(path, "rb") as f:
        data = f.read()

    # Load the state dict from the read data
    return th.load(io.BytesIO(data), **kwargs)