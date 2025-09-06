from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation
from viva.utils.path_utils import Pathable


def _load_and_transform(path: Pathable,
                        transforms: Optional[List[BaseLandmarkAugmentation]] = None) -> Optional[FaceLandmarkSeries]:
    path = Path(path)
    series = FaceLandmarkSeries.load(path)

    if series is None:
        return None

    if len(series.samples) == 0:
        return None

    if transforms is None:
        return series

    # add source path
    series._metadata_path = path

    # apply transforms
    x = series.samples
    y = series.speaking_labels
    for transform in transforms:
        x, y = transform(x, y, series, 0, series.sample_count)
    series.samples = x
    series.speaking_labels = y
    return series


def load_face_landmark_series_in_parallel(paths: List[Pathable],
                                          transforms: Optional[List[BaseLandmarkAugmentation]] = None,
                                          max_threads: int = 4,
                                          show_progress: bool = True) -> List[FaceLandmarkSeries]:
    series = []
    with ThreadPoolExecutor(max_threads) as executor:
        # Use a thread-safe progress bar
        futures = list(tqdm(executor.map(_load_and_transform, paths, repeat(transforms)),
                            desc="loading series", total=len(paths), disable=not show_progress))
        series.extend(futures)

    return series
