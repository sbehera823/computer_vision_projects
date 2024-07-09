"""
Contains functions with different data transforms
"""

from typing import Sequence, Tuple

import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import RandomHorizontalFlip, ColorJitter, RandomCrop, RandomRotation


def get_fundamental_transforms(inp_size: Tuple[int, int]) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fundamental_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    fundamental_transforms = transforms.Compose([
        transforms.Resize(inp_size),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485],  
                             std=[0.229])
    ])

    return fundamental_transforms

    ###########################################################################
    # Student code ends
    ###########################################################################

def get_fundamental_augmentation_transforms(
    inp_size: Tuple[int, int]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    aug_transforms = transforms.Compose([
        RandomHorizontalFlip(),  
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        RandomRotation(degrees=15),  
        transforms.RandomResizedCrop(size=inp_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        Resize(inp_size),
        ToTensor(), 
        Normalize(mean=[0.485], std=[0.229]) 
    ])
    return aug_transforms

    ###########################################################################
    # Student code end
    ###########################################################################

def get_fundamental_normalization_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model alomg with
    normalization.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fund_norm_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    fund_norm_transforms = transforms.Compose([
        transforms.Resize(inp_size),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=pixel_mean, std=pixel_std)  
    ])

    return fund_norm_transforms

    ###########################################################################
    # Student code ends
    ###########################################################################


def get_all_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set,
    along with normalization. This should just be your previous method + normalization.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    all_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    all_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=inp_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),  

        transforms.Resize(inp_size),
        transforms.ToTensor(),

        transforms.Normalize(mean=pixel_mean, std=pixel_std)
    ])

    return all_transforms

    ###########################################################################
    # Student code ends
    ###########################################################################
