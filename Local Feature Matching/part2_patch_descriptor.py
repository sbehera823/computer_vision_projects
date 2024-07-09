#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    X, Y = X.astype(int), Y.astype(int)
    fvs = np.zeros((len(X), feature_width**2))
    for i in range(len(X)):
        x1 = X[i] - feature_width // 2 + 1
        y1 = Y[i] - feature_width // 2 + 1
        x2 = X[i] + feature_width // 2 + 1
        y2 = Y[i] + feature_width // 2 + 1
        patch = image_bw[max(y1, 0):min(y2, image_bw.shape[0]), max(x1, 0):min(x2, image_bw.shape[1])]
        resized_patch = np.resize(patch, (1, feature_width**2))
        norm = np.linalg.norm(resized_patch)
        if norm:
            w = resized_patch / norm
            if np.any(w):
                fvs[i] = w
            else:
                fvs[i] = resized_patch
        else:
            fvs[i] = resized_patch

    return fvs

    # raise NotImplementedError('`compute_normalized_patch_descriptors` ' +
    #     'function in`part2_patch_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
