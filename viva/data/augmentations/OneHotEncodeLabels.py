from typing import Tuple

import numpy as np

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation


class OneHotEncodeLabels(BaseLandmarkAugmentation):
    """
    Convert binary labels in the y tensor (True/False or 1/0) to one-hot encoded vectors.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray,
                 series: FaceLandmarkSeries, start_index: int, end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure y is binary and convert to integers if necessary
        y_binary = y.astype(int)
        # Create one-hot encoding: [0] -> [1, 0], [1] -> [0, 1]
        y_one_hot = np.eye(2)[y_binary]
        return x, y_one_hot
