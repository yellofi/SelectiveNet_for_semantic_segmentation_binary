import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .laplacian import *

def load_model():
    model = Laplacian()
    return model
