from typing import Tuple

import numpy as np

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation


class FlattenLandmarks(BaseLandmarkAugmentation):
    """
    Flatten the landmark data from (b, n, 3) to (b, n * 3).
    """

    def __call__(self, x: np.ndarray, y: np.ndarray,
                 series: FaceLandmarkSeries, start_index: int, end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        x = x.reshape(x.shape[0], -1)  # Flatten the last two dimensions
        return x, y
