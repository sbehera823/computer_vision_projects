import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    pixel_values = []  
    
    for category_subdir in os.listdir(dir_name):
        category_path = os.path.join(dir_name, category_subdir)
        if os.path.isdir(category_path):
            for image_subdir in os.listdir(category_path):
                image_subdir_path = os.path.join(category_path, image_subdir)
                if os.path.isdir(image_subdir_path):
                    for img_path in glob.glob(os.path.join(image_subdir_path, '*.jpg')): 
                        with Image.open(img_path) as img:
                            img_gray = img.convert('L')
                            img_array = np.array(img_gray) / 255.0
                            pixel_values.extend(img_array.flatten())

    if not pixel_values:  
        print("No pixels were loaded. Please check the image paths and formats.")
        return float('nan'), float('nan')  

    pixel_values = np.array(pixel_values)

    mean = np.mean(pixel_values)
    std = np.std(pixel_values)

    return mean, std

    ############################################################################
    # Student code end
    ############################################################################
