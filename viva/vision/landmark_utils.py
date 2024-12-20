import numpy as np
from visiongraph import vg


def normalize_landmarks(landmarks: np.ndarray, transformation_matrix: np.ndarray,
                        origin_landmark_index: int = vg.BlazeFaceMesh.NOSE_INDEX) -> np.ndarray:
    """
    Normalize 3D landmark points to a canonical space using the inverse of
    the specified transformation matrix.

    :params landmarks: Landmarks in shape (n, m>=3)
    :params transformation_matrix: Transformation matrix of the landmarks.
    :params origin_landmark_index: Index of the origin landmark for the normalisation.
    :returns: A NumPy array containing the normalized 3D landmarks.
    """
    inv_matrix = np.linalg.inv(transformation_matrix)
    inv_rotation, inv_translation, inv_scale = vg.decompose_transformation_matrix(inv_matrix)

    vertices = np.array([[e[0], 1 - e[1], -e[2]] for e in landmarks], dtype=np.float32)
    canonical_vertices = vertices @ inv_rotation.T

    origin = canonical_vertices[origin_landmark_index]

    # Translate vertices to center the origin
    canonical_vertices -= origin

    # Normalize vertices to be between -1 and +1 based on the maximum distance from the origin
    max_distance = np.max(np.linalg.norm(canonical_vertices, axis=1))
    normalized_vertices = canonical_vertices / max_distance

    return normalized_vertices


def normalize_landmark_batch(landmarks: np.ndarray, transformation_matrices: np.ndarray,
                             origin_landmark_index: int = vg.BlazeFaceMesh.NOSE_INDEX):
    return np.array([normalize_landmarks(lms, t, origin_landmark_index)
                     for lms, t in zip(landmarks, transformation_matrices)])
