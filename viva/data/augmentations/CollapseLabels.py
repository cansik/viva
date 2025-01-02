from typing import Tuple

import numpy as np

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation


class CollapseLabels(BaseLandmarkAugmentation):
    """
    Collapse (n, 1) boolean labels in the y tensor to a single-dimensional array.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray,
                 series: FaceLandmarkSeries, start_index: int, end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        # todo: maybe use average label (with sum)
        return x, y[0]
