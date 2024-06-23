import numpy as np

def extractRt(E):
    """
    Extracts the rotation and translation matrices from the essential matrix.

    Parameters
    ----------
    E : np.array
        The essential matrix.

    Returns
    -------
    Rt : np.array
        Concatenation of rotation matrix and translation vector.
    """
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(E)

    # Ensure U and Vt are proper rotation matrices
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0

    # Choose the R whose diagonal values are closer to 1
    R = np.dot(U, np.dot(W, Vt))
    if R.trace() < 0:
        R = np.dot(U, np.dot(W.T, Vt))
    t = U[:, 2]

    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    
    return Rt

def homogeneous_coord(x):
    """
    Converts the points to homogeneous coordinates.

    Parameters
    ----------
    x : np.array
        The points to convert.

    Returns
    -------
    np.array
        The points in homogeneous coordinates.
    """
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def normalize_points(pts, Kinv):

        normalized_points = np.dot(Kinv, homogeneous_coord(pts).T).T[:, 0:2]
        return normalized_points

def denormalize_point(pt, K):
        ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
        denormalized_point =  int(round(ret[0])), int(round(ret[1]))
        return denormalized_point