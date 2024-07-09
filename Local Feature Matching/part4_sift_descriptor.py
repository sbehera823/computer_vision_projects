#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from vision.part1_harris_corner import compute_image_gradients
from torch import nn
from typing import Tuple


"""
Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Your implementation will not exactly match the SIFT reference. For example,
we will be excluding scale and rotation invariance.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells. 
"""


def get_magnitudes_and_orientations(
    Ix: np.ndarray,
    Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location.

    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        magnitudes: A numpy array of shape (m,n), representing magnitudes of
            the gradients at each pixel location
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    magnitudes = np.sqrt(Ix**2 + Iy**2)
    orientations = np.arctan2(Iy, Ix)
    return magnitudes, orientations

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################



def get_gradient_histogram_vec_from_patch(
    window_magnitudes: np.ndarray,
    window_orientations: np.ndarray
) -> np.ndarray:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms.

    Key properties to implement:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the terminology
        used in the feature literature to describe the spatial bins where
        gradient distributions will be described. The grid will extend
        feature_width/2 - 1 to the left of the "center", and feature_width/2 to
        the right. The same applies to above and below, respectively. 
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be
        added to the feature vector left to right then row by row (reading
        order).

    Do not normalize the histogram here to unit norm -- preserve the histogram
    values. A useful function to look at would be np.histogram.

    Args:
        window_magnitudes: (16,16) array representing gradient magnitudes of the
            patch
        window_orientations: (16,16) array representing gradient orientations of
            the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    wgh = np.zeros((0, 1), dtype=np.float32)
    bin_edges = np.linspace(-np.pi, np.pi, 9) - 1e-5
    for row in range(4):
        for col in range(4):
            cell_mags = window_magnitudes[row*4:(row+1)*4, col*4:(col+1)*4]
            cell_oris = window_orientations[row*4:(row+1)*4, col*4:(col+1)*4]
            cell_hist = np.histogram(cell_oris, bins=bin_edges, weights=cell_mags)[0].reshape(-1, 1)
            wgh = np.concatenate([wgh, cell_hist], axis=0)

    return wgh
 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################



def get_feat_vec(
    c: float,
    r: float,
    magnitudes,
    orientations,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.


    Your (baseline) descriptor should have:
    (1) Each feature should be normalized to unit length.
    (2) Each feature should be raised to the 1/2 power, i.e. square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
    
    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions.
    The autograder will only check for each gradient contributing to a single bin.
    
    Args:
        c: a float, the column (x-coordinate) of the interest point
        r: A float, the row (y-coordinate) of the interest point
        magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
        orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fv: A numpy array of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    fv = []#placeholder
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    pad_before = (feature_width - 1) // 2
    pad_after = feature_width // 2
    padded_mags = np.pad(magnitudes, ((pad_before, pad_after), (pad_before, pad_after)), mode='constant')
    padded_oris = np.pad(orientations, ((pad_before, pad_after), (pad_before, pad_after)), mode='constant')
    window_mag = padded_mags[r:r + feature_width, c:c + feature_width]
    window_ori = padded_oris[r:r + feature_width, c:c + feature_width]
    feature_vector = get_gradient_histogram_vec_from_patch(window_mag, window_ori)

    if np.linalg.norm(feature_vector) != 0:
        fv = np.sqrt(feature_vector / np.linalg.norm(feature_vector))
    else:
        fv = feature_vector

    return fv

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


def get_SIFT_descriptors(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the 128-d SIFT features computed at each of the input
    points. Implement the more effective SIFT descriptor (see Szeliski 7.1.2 or
    the original publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A numpy array of shape (m,n), the image
        X: A numpy array of shape (k,), the x-coordinates of interest points
        Y: A numpy array of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e.,
            every cell of your local SIFT-like feature will have an integer
            width and height). This is the initial window size we examine
            around each keypoint.
    Returns:
        fvs: A numpy array of shape (k, feat_dim) representing all feature
            vectors. "feat_dim" is the feature_dimensionality (e.g., 128 for
            standard SIFT). These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    Ix, Iy = compute_image_gradients(image_bw)
    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)

    fvs = np.zeros((len(X), 128)) 
    for i, (c, r) in enumerate(zip(X, Y)):
        fv = get_feat_vec(c, r, magnitudes, orientations, feature_width).flatten()
        fvs[i, :] = fv

    return fvs

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def rotate_image(
    image: np.ndarray,
    angle: int
) -> np.ndarray :
    """
    Rotate an image by a given angle around its center.

    Args:
    image: numpy array of the image to be rotated
    angle: the angle by which to rotate the image (in degrees)

    Returns:
    Rotated Image as a numpy array

    Note:
    1)Convert the rotation angle from degrees to radians
    2)Find the center of the image (around which the rotation will occur)
    3)Define the rotation matrix for rotating around the image center
    4)Rotation matrix can be [[cos, -sin, center_x*(1-cos)+center_y*sin],
                              [sin,  cos, center_y*(1-cos)-center_x*sin],
                              [0,    0,   1,]]
    5)Apply affine transformation
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    angle_rad = np.radians(-angle)
    height, width = image.shape[:2]
    cx, cy = (width / 2) - 0.5, (height / 2) - 0.5
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_mat = np.array([
        [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy],
        [sin_a, cos_a, (1 - cos_a) * cy - sin_a * cx]
    ])    
    rotated = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            xy_prime = np.dot(rot_mat, [x, y, 1])
            x_prime, y_prime = xy_prime[:2]
            x_p, y_p = int(round(x_prime)), int(round(y_prime))            
            if 0 <= x_p < width and 0 <= y_p < height:
                rotated[y, x] = image[y_p, x_p]
                
    return rotated
    # raise NotImplementedError('`rotate_image` function in ' +
    #     '`part4_sift_descriptor.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

def crop_center(image, new_width, new_height):
    """
    Crop the central part of an image to the specified dimensions.

    Args:
    image: The image to crop.
    new_width: The target width of the cropped image.
    new_height: The target height of the cropped image.

    Returns:
    cropped image as a numpy array
    """
    height, width = image.shape[:2]
    start_x = width // 2 - new_width // 2
    start_y = height // 2 - new_height // 2
    cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]
    return cropped_image

def get_correlation_coeff(
    v1: np.ndarray,
    v2: np.ndarray
) -> float:
    """
    Compute the correlation coefficient between two vectors v1 and v2. Refer to the notebook for the formula.
    Args:
    v1: the first vector
    v2: the second vector
    Returns:
    The scalar correlation coefficient between the two vectors
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                         
    #############################################################################

    numerator = np.dot(v1.T, v2)
    denominator = np.sqrt(np.dot(v1.T, v1) * np.dot(v2.T, v2))
    
    if denominator == 0:
        return 0.0
    
    rho = numerator / denominator
    return rho

    # raise NotImplementedError('`get_correlation_coeff` function in ' +
    #     '`part4_sift_descriptor.py` needs to be implemented')
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
def get_intensity_based_matches(
    image1: np.ndarray,
    image2: np.ndarray, 
    window_size=64,
    stride=128
) -> np.ndarray:
    """
    Compute intensity-based matches between image1 and image2. For each patch in image1, obtain the patch in image2 with the maximum correlation coefficient.
    Args:
    image1: the first image
    image2: the second image
    window_size: the size of each patch(window) in the images
    stride: the number of pixels by which each patch is shifted to obtain the next patch
    Returns:
    A 3-D numpy array of the form: [[x1, y1],[x2,y2]], where
    x1: x-coordinate of top-left corner of patch in image1
    y1: y-coordinate of top-left corner of patch in image1
    x2: x-coordinate of top-left corner of matching patch in image2
    y2: y-coordinate of top-left corner of matching patch in image2
    """

    # reshaping images to the same dimensions
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])
    image1 = image1[:min_height, :min_width,:]
    image2 = image2[:min_height, :min_width,:]

    #normalizing images for pixel values to be between 0 and 1
    image1 = (image1-np.min(image1))/(np.max(image1)-np.min(image1)) 
    image2 = (image2-np.min(image2))/(np.max(image2)-np.min(image2)) 
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                         
    #############################################################################
    matches = []
    for y1 in range(0, image1.shape[0] - window_size + 1, stride):
        for x1 in range(0, image1.shape[1] - window_size + 1, stride):
            patch1 = image1[y1:y1 + window_size, x1:x1 + window_size]
            max_corr = -1
            match_coord = (0, 0)
            for y2 in range(0, image2.shape[0] - window_size + 1, stride):
                for x2 in range(0, image2.shape[1] - window_size + 1, stride):
                    patch2 = image2[y2:y2 + window_size, x2:x2 + window_size]
                    corr = get_correlation_coeff(patch1.flatten(), patch2.flatten())
                    if corr > max_corr:
                        max_corr = corr
                        match_coord = (x2, y2)
            matches.append([[x1, y1], list(match_coord)])

    return np.array(matches)

    # raise NotImplementedError('`get_intensity_based_matches` function in ' +
    #     '`part4_sift_descriptor.py` needs to be implemented')
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################




