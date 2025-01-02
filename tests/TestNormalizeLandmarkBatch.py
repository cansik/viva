import unittest

import numpy as np

from viva.vision.landmark_utils import normalize_landmark_batch


class TestNormalizeLandmarkBatch(unittest.TestCase):
    def setUp(self):
        # Create a simple batch of landmarks and transformation matrices
        self.landmarks = np.array([
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]],
        ])  # Shape: (2, 4, 3)

        self.transformation_matrices = np.array([
            np.eye(4),
            np.eye(4),
        ])  # Identity transformation matrices, Shape: (2, 4, 4)

    def test_invariance_to_translation(self):
        # Translate the landmarks by a random vector
        translation = np.array([[1, -2, 3]])  # Shape: (1, 3)
        translated_landmarks = self.landmarks + translation

        normalized_original = normalize_landmark_batch(
            self.landmarks, self.transformation_matrices
        )
        normalized_translated = normalize_landmark_batch(
            translated_landmarks, self.transformation_matrices
        )

        np.testing.assert_almost_equal(
            normalized_original, normalized_translated,
            err_msg="Normalization failed to be invariant to translation."
        )

    def test_invariance_to_scaling(self):
        # Scale the landmarks by a random factor
        scale_factor = 2.0
        scaled_landmarks = self.landmarks * scale_factor

        normalized_original = normalize_landmark_batch(
            self.landmarks, self.transformation_matrices
        )
        normalized_scaled = normalize_landmark_batch(
            scaled_landmarks, self.transformation_matrices
        )

        np.testing.assert_almost_equal(
            normalized_original, normalized_scaled,
            err_msg="Normalization failed to be invariant to scaling."
        )

    def test_normalized_landmarks_have_unit_max_distance(self):
        # Ensure that the normalized landmarks have a maximum distance of 1
        normalized_landmarks = normalize_landmark_batch(
            self.landmarks, self.transformation_matrices
        )

        max_distances = np.linalg.norm(normalized_landmarks, axis=2).max(axis=1)
        np.testing.assert_almost_equal(
            max_distances, np.ones_like(max_distances),
            err_msg="Normalized landmarks do not have unit maximum distance."
        )

    def test_origin_landmark_is_centered(self):
        # Ensure the origin landmark is centered at [0, 0, 0]
        origin_index = 0  # Use the first landmark as the origin
        normalized_landmarks = normalize_landmark_batch(
            self.landmarks, self.transformation_matrices, origin_landmark_index=origin_index
        )

        origins = normalized_landmarks[:, origin_index, :]
        np.testing.assert_almost_equal(
            origins, np.zeros_like(origins),
            err_msg="Origin landmark is not centered at [0, 0, 0]."
        )


if __name__ == "__main__":
    unittest.main()
