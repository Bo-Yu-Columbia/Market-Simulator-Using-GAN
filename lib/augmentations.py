"""
This file provides utilities for data augmentation and the computation of path signatures. 
These augmentations enhance the capability of capturing important features in the first components of the signature, 
particularly in the context of sequential or time-series data.

The file contains implementations of different types of augmentations, such as scaling, concatenation, cumsum, 
adding lags and lead-lag transformation. Each augmentation is implemented as a dataclass with an apply method, which 
applies the augmentation to a given tensor. These augmentations can be combined in a specific order to create a 
more complex augmentation pipeline.

The file also provides utilities for computing the signature of a path, which is a method to transform sequence data 
into a format that captures both the values and the order of data points. This is particularly useful in machine learning 
tasks involving time-series data.

Finally, the file defines a SignatureConfig dataclass for specifying the configuration of the signature computation, 
including the augmentations to apply, the depth of the signature, and whether or not to use a basepoint. There is also 
a helper function get_standard_augmentation that returns a standard pipeline of augmentations.
"""
from dataclasses import dataclass
from typing import List, Tuple

import signatory
import torch

__all__ = ['AddLags', 'Concat', 'Cumsum', 'LeadLag', 'Scale']


def get_time_vector(size: int, length: int) -> torch.Tensor:
    # Define a function to create a time vector tensor

    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def lead_lag_transform(x: torch.Tensor) -> torch.Tensor:
    # The `lead_lag_transform` function applies the lead-lag transformation to a given tensor. 

    # The lead-lag transform is a method used in time-series analysis to create a new series from an 
    # existing one, where the new series is a lagged version of the original one.


    x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
    x_ll = torch.cat([x_rep[:, :-1], x_rep[:, 1:]], dim=2)
    return x_ll


def lead_lag_transform_with_time(x: torch.Tensor) -> torch.Tensor:
    # The `lead_lag_transform_with_time` function applies the lead-lag transformation and also includes 
    # the time vector in the transformed tensor.

    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    t_rep = torch.repeat_interleave(t, repeats=3, dim=1)
    x_rep = torch.repeat_interleave(x, repeats=3, dim=1)
    x_ll = torch.cat([
        t_rep[:, 0:-2],
        x_rep[:, 1:-1],
        x_rep[:, 2:],
    ], dim=2)
    return x_ll


def cat_lags(x: torch.Tensor, m: int) -> torch.Tensor:
    # The `cat_lags` function is used to add lags to a tensor. Lags are simply delayed versions of the original series, 
    # and can provide useful context for time-series prediction tasks.

    q = x.shape[1]
    assert q >= m, 'Lift cannot be performed. q < m : (%s < %s)' % (q, m)
    x_lifted = list()
    for i in range(m):
        x_lifted.append(x[:, i:i + m])
    return torch.cat(x_lifted, dim=-1)


@dataclass
class BaseAugmentation:
    # No base augmentation implemented
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        # The `apply` method of each augmentation class applies the specific augmentation to a given tensor.

        raise NotImplementedError('Needs to be implemented by child.')

# Below are different types of augmentation methods and the helper function to apply the augmentations

@dataclass
class Scale(BaseAugmentation):
    scale: float = 1

    def apply(self, x: torch.Tensor):
        return self.scale * x


@dataclass
class Concat(BaseAugmentation):

    @staticmethod
    def apply(x: torch.Tensor, y: torch.Tensor):
        return torch.cat([x, y], dim=-1)


@dataclass
class Cumsum(BaseAugmentation):
    dim: int = 1

    def apply(self, x: torch.Tensor):
        return x.cumsum(dim=self.dim)


@dataclass
class AddLags(BaseAugmentation):
    m: int = 2

    def apply(self, x: torch.Tensor):
        return cat_lags(x, self.m)


@dataclass
class LeadLag(BaseAugmentation):
    with_time: bool = False

    def apply(self, x: torch.Tensor):
        if self.with_time:
            return lead_lag_transform_with_time(x)
        else:
            return lead_lag_transform(x)


def _apply_augmentation(x: torch.Tensor, y: torch.Tensor, augmentation) -> Tuple[torch.Tensor, torch.Tensor]:
    # The `_apply_augmentation` function is a helper function that applies a given augmentation to a tensor.

    if type(augmentation).__name__ == 'Concat':  # todo
        return y, augmentation.apply(x, y)
    else:
        return y, augmentation.apply(y)


def apply_augmentations(x: torch.Tensor, augmentations: Tuple) -> torch.Tensor:
    # The `apply_augmentations` function applies a sequence of augmentations to a tensor.

    y = x
    for augmentation in augmentations:
        x, y = _apply_augmentation(x, y, augmentation)
    return y


@dataclass
class SignatureConfig:
    augmentations: Tuple
    depth: int
    basepoint: bool = False


def augment_path_and_compute_signatures(x: torch.Tensor, config: SignatureConfig) -> torch.Tensor:
    # The `augment_path_and_compute_signatures` function applies a sequence of augmentations to a tensor and then computes its path signature.

    y = apply_augmentations(x, config.augmentations)
    return signatory.signature(y, config.depth, basepoint=config.basepoint)


def get_standard_augmentation(scale: float) -> Tuple:
    # The `get_standard_augmentation` function returns a standard sequence of augmentations.
    
    return tuple([Scale(scale), Cumsum(), Concat(), AddLags(m=2), LeadLag(with_time=False)])
