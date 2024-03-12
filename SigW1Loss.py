from typing import Tuple, Optional,List
from dataclasses import dataclass


import signatory
import torch
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import math


def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)

@dataclass
class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('Needs to be implemented by child.')


@dataclass
class Scale(BaseAugmentation):
    scale: float = 1
    dim: int = None

    def apply(self, x: torch.Tensor):
        if self.dim == None:
            return self.scale * x
        else:
            x[...,self.dim] = self.scale * x[...,self.dim]
            return x


@dataclass
class AddTime(BaseAugmentation):

    def apply(x: torch.Tensor):
        t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
        return torch.cat([t, x], dim=-1)


@dataclass
class Basepoint(BaseAugmentation):

    def apply(self, x: torch.Tensor):
        basepoint = torch.zeros(x.shape[0], 1, x.shape[2]).to(x.device)
        return torch.cat([basepoint, x], dim=1)

def apply_augmentations(x: torch.Tensor, augmentations: Tuple) -> torch.Tensor:
    y = x.clone()
    for augmentation in augmentations:
        y = augmentation.apply(y)
    return y

def compute_expected_signature(x_path, depth: int, augmentations: Tuple, normalise: bool = True):
    x_path_augmented = apply_augmentations(x_path, augmentations)
    expected_signature = signatory.signature(x_path_augmented, depth=depth).mean(0)
    dim = x_path_augmented.shape[2]
    count = 0
    if normalise:
        for i in range(depth):
            expected_signature[count:count + dim**(i+1)] = expected_signature[count:count + dim**(i+1)] * math.factorial(i+1)
            count = count + dim**(i+1)
    return expected_signature

def compute_signature(x_path, depth: int, augmentations=None, normalise: bool = True):
    if augmentations == None:
        x_path_augmented = x_path
    else:
        x_path_augmented = apply_augmentations(x_path, augmentations)
    print(x_path_augmented.shape)
    print(x_path.shape)
    signatures = signatory.signature(x_path_augmented, depth=depth)
    dim = x_path_augmented.shape[2]
    count = 0
    if normalise:
        for i in range(depth):
            signatures[:,count:count + dim**(i+1)] = signatures[:,count:count + dim**(i+1)] * math.factorial(i+1)
            count = count + dim**(i+1)
    return signatures


def rmse(x, y):
    return (x - y).pow(2).mean(0).sum().sqrt()

def masked_rmse(x, y, mask_rate, device):
    mask = torch.FloatTensor(x.shape[0]).to(device).uniform_() > mask_rate
    mask = mask.int()
    return ((x - y).pow(2) * mask).mean().sqrt()


class SigW1Metric:
    def __init__(self, depth: int, x_real: torch.Tensor, mask_rate:float, augmentations: Optional[Tuple] = (), normalise: bool = True):
        if len(x_real.shape) == 2:
            x_real = x_real[:,:,None]

        self.augmentations = augmentations
        self.depth = depth
        self.n_lags = x_real.shape[1]
        self.mask_rate = mask_rate

        self.normalise = normalise
        self.signatures_mu = compute_signature(x_real, depth, augmentations, normalise)
        

    def __call__(self, x_path_nu: torch.Tensor):
        """ Computes the SigW1 metric."""
        if len(x_path_nu.shape) == 2:
            x_path_nu = x_path_nu[:,:,None]
        device = x_path_nu.device
        batch_size = x_path_nu.shape[0]

        signatures_nu = compute_signature(x_path_nu, self.depth, self.augmentations, self.normalise)
        loss = rmse(self.signatures_mu.to(device), signatures_nu)
        #loss = masked_rmse(self.signatures_mu.to(
        #    device), signatures_nu, self.mask_rate, device)
        return loss
