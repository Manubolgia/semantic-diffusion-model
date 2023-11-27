"""
Helpers for distributed training.
"""

import os
import torch as th

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
