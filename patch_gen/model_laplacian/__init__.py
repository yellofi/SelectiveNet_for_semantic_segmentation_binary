import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .laplacian import *

def load_model():
    model = Laplacian()
    return model

def make_patch_(patch):
    """
    corresponding smaller sized gray-scale patch for blurrity check 
    """
    patch = np.array(patch)
    patch = patch.astype(np.float32)
    patch = np.expand_dims(patch, axis=0)
    patch = np.ascontiguousarray(patch, dtype=np.float32)
    return patch