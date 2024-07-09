"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    points_mean = np.mean(points, axis=0)
    points_std = np.std(points, axis=0, ddof=1)
    
    scale_factors = 1.07 / points_std
    
    normalization_matrix = np.diag(scale_factors.tolist() + [1])
    normalization_matrix[0, 2] = -points_mean[0] * scale_factors[0]
    normalization_matrix[1, 2] = -points_mean[1] * scale_factors[1]
    
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    
    normalized_homogeneous_points = normalization_matrix.dot(points_homogeneous.T)
    
    normalized_points = normalized_homogeneous_points[:2, :].T

    return normalized_points, normalization_matrix

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################



def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = np.dot(T_b.T, np.dot(F_norm, T_a))

    return F_orig

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################



def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    norm_pts_a, transform_A = normalize_points(points_a)
    norm_pts_b, transform_B = normalize_points(points_b)

    matrix_A = np.zeros((len(points_a), 9))
    for idx, (pt_a, pt_b) in enumerate(zip(norm_pts_a, norm_pts_b)):
        xa, ya = pt_a
        xb, yb = pt_b
        matrix_A[idx] = [xb * xa, xb * ya, xb, yb * xa, yb * ya, yb, xa, ya, 1]

    _, _, V = np.linalg.svd(matrix_A)
    F_temp = V[-1].reshape(3, 3)

    Uf, Df, Vf = np.linalg.svd(F_temp)
    Df[2] = 0  
    F_corrected = Uf @ np.diag(Df) @ Vf

    F = unnormalize_F(F_corrected, transform_A, transform_B)

    return F

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
