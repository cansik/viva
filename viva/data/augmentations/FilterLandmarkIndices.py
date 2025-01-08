from typing import Tuple, Sequence

import numpy as np

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation


class FilterLandmarkIndices(BaseLandmarkAugmentation):
    """
    Filter landmarks to retain only specific indices.
    """

    def __init__(self, landmark_indices: Sequence[int]):
        self.landmark_indices: np.ndarray = np.array(list(landmark_indices), dtype=np.uint32)

    def __call__(self, x: np.ndarray, y: np.ndarray,
                 series: FaceLandmarkSeries, start_index: int, end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        x = x[:, self.landmark_indices]
        return x, y
