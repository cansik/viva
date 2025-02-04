import random
from typing import Tuple

import numpy as np

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation


class RandomNoiseAugmentation(BaseLandmarkAugmentation):
    def __init__(self, noise_std: float = 0.1, clip: bool = True, probability: float = 0.5):
        """
        Initialize the augmentation.

        :param noise_std: Standard deviation of the Gaussian noise to be added.
                          Since the data is normalized between -1 and 1, choose an appropriate value.
        :param clip: If True, clip the augmented values to remain in [-1, 1].
        :param probability: Probability (0-1) if the augmentation is applied.
        """
        self.noise_std = noise_std
        self.clip = clip
        self.probability = probability

    def __call__(self, x: np.ndarray, y: np.ndarray,
                 series: FaceLandmarkSeries, start_index: int, end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random Gaussian noise to the landmark data.

        :param x: Landmark data (numpy array), expected to be normalized between -1 and 1.
        :param y: Labels corresponding to the landmarks.
        :param series: FaceLandmarkSeries containing all relevant series information.
        :param start_index: Start index of the current sequence inside the series.
        :param end_index: End index of the current sequence inside the series.
        :return: Tuple of augmented landmark data (x) and labels (y).
        """
        if random.random() > self.probability:
            return x, y

        # Generate noise with the same shape as the input landmark data
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=x.shape)

        # Add noise to the landmarks
        x_aug = x + noise

        # Optionally clip the values to ensure they remain within the [-1, 1] range.
        if self.clip:
            x_aug = np.clip(x_aug, -1, 1)

        return x_aug, y
