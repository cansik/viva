from typing import Tuple

import numpy as np
from visiongraph import vg

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation
from viva.vision.landmark_utils import normalize_landmark_batch


class NormalizeLandmarks(BaseLandmarkAugmentation):
    """
    Normalize landmarks using transformation matrices and an origin index.
    """

    def __init__(self, normalize_origin_index: int = vg.BlazeFaceMesh.NOSE_INDEX):
        self.normalize_origin_index = normalize_origin_index

    def __call__(self, x: np.ndarray, y: np.ndarray,
                 series: FaceLandmarkSeries, start_index: int, end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        transform_matrices = series.transforms[start_index:end_index]
        x = normalize_landmark_batch(x, transform_matrices, self.normalize_origin_index)
        return x, y
