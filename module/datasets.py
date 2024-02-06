from typing import Union, List

import numpy as np
import cv2

from torchvision import datasets
from torch.utils.data import Dataset

def get_dataset(name: str, **kwargs):
    if name == "mnist":
        return datasets.MNIST(**kwargs)
    elif name == "cifar10":
        return datasets.CIFAR10(**kwargs)
    else:
        raise ValueError("Incorrect Name")