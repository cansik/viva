from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries


class BaseLandmarkAugmentation(ABC):
    """
    Abstract base class for data augmentations applied to face landmarks.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray,
                 series: FaceLandmarkSeries, start_index: int, end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the augmentation.

        :param x: Landmark data, typically a numpy array.
        :param y: Labels corresponding to the landmarks, typically a numpy array.
        :param series: FaceLandmarkSeries containing all relevant series information.
        :param start_index: Start index of the current sequence inside the series.
        :param end_index: End index of the current sequence inside the series.
        :return: Tuple of augmented landmarks (x) and labels (y).
        """
        pass
