import numpy as np
from visiongraph import vg


def normalize_landmarks(landmarks: np.ndarray, transformation_matrix: np.ndarray,
                        origin_landmark_index: int = vg.BlazeFaceMesh.NOSE_INDEX) -> np.ndarray:
    """
    Normalize 3D landmark points to a canonical space using the inverse of
    the specified transformation matrix.

    :params landmarks: Landmarks in shape (n, m>=3)
    :params transformation_matrix: Transformation matrix of the landmarks.
    :params origin_landmark_index: Index of the origin landmark for the normalization.
    :returns: A NumPy array containing the normalized 3D landmarks.
    """
    # Decompose the transformation matrix once
    inv_matrix = np.linalg.inv(transformation_matrix)
    inv_rotation, inv_translation, inv_scale = vg.decompose_transformation_matrix(inv_matrix)

    # Normalize landmarks in a vectorized manner
    vertices = np.column_stack((landmarks[:, 0], 1 - landmarks[:, 1], -landmarks[:, 2]))
    canonical_vertices = vertices @ inv_rotation.T

    # Translate vertices to center the origin landmark
    origin = canonical_vertices[origin_landmark_index]
    canonical_vertices -= origin

    # Normalize vertices to be between -1 and +1 based on the maximum distance from the origin
    max_distance = np.linalg.norm(canonical_vertices, axis=1).max()
    normalized_vertices = canonical_vertices / max_distance

    return normalized_vertices


def normalize_landmark_batch(
        landmarks: np.ndarray,
        transformation_matrices: np.ndarray,
        origin_landmark_index: int = vg.BlazeFaceMesh.NOSE_INDEX
) -> np.ndarray:
    """
    Batch normalize 3D landmark points to a canonical space (optimized).

    :params landmarks: Batch of landmarks in shape (b, n, m>=3)
    :params transformation_matrices: Batch of transformation matrices in shape (b, 4, 4)
    :params origin_landmark_index: Index of the origin landmark for the normalization.
    :returns: A NumPy array containing the batch-normalized 3D landmarks.
    """
    # Precompute inverse transformations for all matrices
    inv_matrices = np.linalg.inv(transformation_matrices)  # (b, 4, 4)
    inv_rotations = inv_matrices[:, :3, :3]  # Extract the inverse rotation matrices (b, 3, 3)

    # Ensure landmarks are in the correct shape
    if landmarks.shape[-1] > 3:
        landmarks = landmarks[..., :3]  # Discard extra dimensions if present

    # Reshape landmarks to apply rotation in a vectorized manner
    vertices = np.empty_like(landmarks)
    vertices[:, :, 0] = landmarks[:, :, 0]
    vertices[:, :, 1] = 1 - landmarks[:, :, 1]
    vertices[:, :, 2] = -landmarks[:, :, 2]

    # Perform matrix multiplication: (b, n, 3) x (b, 3, 3)
    canonical_vertices = np.matmul(vertices, inv_rotations.transpose(0, 2, 1))

    # Translate to center the origin landmark for each batch
    origins = canonical_vertices[:, origin_landmark_index, :]  # (b, 3)
    canonical_vertices -= origins[:, np.newaxis, :]  # Broadcasting subtraction

    # Normalize by the maximum distance in each batch
    max_distances = np.linalg.norm(canonical_vertices, axis=2).max(axis=1)  # (b,)
    normalized_vertices = canonical_vertices / max_distances[:, np.newaxis, np.newaxis]

    return normalized_vertices
