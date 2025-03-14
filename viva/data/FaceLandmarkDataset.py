from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from viva.data import zarr_io
from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation
from viva.data.face_landmark_io import load_face_landmark_series_in_parallel
from viva.utils.RangeMap import RangeMap, RangeResult
from viva.utils.path_utils import Pathable, get_files


class FaceLandmarkDataset(Dataset):

    def __init__(self,
                 data_path: Optional[Pathable] = None,
                 metadata_paths: Optional[List[Pathable]] = None,
                 block_length: int = 15,
                 stride: int = 1,
                 use_blend_shapes: bool = False,
                 transforms: Optional[List[BaseLandmarkAugmentation]] = None,
                 augmentations: Optional[List[BaseLandmarkAugmentation]] = None):
        super().__init__()
        self.block_length = block_length
        self.stride = stride
        self.use_blend_shapes = use_blend_shapes

        self.data_path: Optional[Path] = Path(data_path) if data_path is not None else None
        self.metadata_paths: List[Path] = self._load_metadata_files() if data_path is not None else metadata_paths

        if self.metadata_paths is None:
            raise ValueError("Please either provide a data path or existing metadata path list!")

        # data
        self.data: List[FaceLandmarkSeries] = []

        # pre-processing
        self.transforms: List[BaseLandmarkAugmentation] = [] if transforms is None else transforms
        self.augmentations: List[BaseLandmarkAugmentation] = [] if augmentations is None else augmentations

        # create index for quick query lookup
        self.data_index = RangeMap[int]()
        self.data_count = 0
        self.create_data_index()

    def create_data_index(self):
        self.data.clear()
        self.data_index.clear()
        self.data_count = 0

        current_index = 0

        # load data in parallel (faster)
        all_series = load_face_landmark_series_in_parallel(self.metadata_paths, self.transforms)

        # filter and create index
        for series_index, (series, metadata_path) in enumerate(list(zip(all_series, self.metadata_paths))):
            series: FaceLandmarkSeries

            if series is None:
                continue

            # calculate max index for the range to have data
            full_length = self.block_length * self.stride
            max_index = (series.sample_count // full_length - 1) * full_length

            if max_index <= 0:
                # todo: add strategies to pad series samples (zero, freeze-last, mirror)
                continue

            self.data_index.add_range(current_index, current_index + max_index, series_index)
            current_index += max_index

        self.data_count = current_index
        self.data = all_series

    def _load_metadata_files(self) -> List[Path]:
        if not self.data_path.is_dir():
            raise Exception("Datapath has to be a directory!")

        return get_files(self.data_path, "*.json", recursive=True)

    def get_series(self, index: int) -> Tuple[FaceLandmarkSeries, RangeResult]:
        range_result = self.data_index[index]
        return self.data[range_result.value], range_result

    def __len__(self) -> int:
        return self.data_count

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        series, range_result = self.get_series(index)

        start_index = index - range_result.start
        end_index = start_index + (self.block_length * self.stride)

        if self.use_blend_shapes:
            x = series.blend_shapes[start_index:end_index]
        else:
            x = series.samples[start_index:end_index]
        y = series.speaking_labels[start_index:end_index].astype(np.float32)

        # apply stride
        x = x[::self.stride]
        y = y[::self.stride]

        # augment landmarks
        x, y = self.apply_augmentations(x, y, series, start_index, end_index)

        return x.astype(np.float32), y.astype(np.float32)

    def apply_augmentations(self, x: np.ndarray, y: np.ndarray, series: FaceLandmarkSeries,
                            start_index: int, end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        for augmentation in self.augmentations:
            x, y = augmentation(x, y, series, start_index, end_index)
        return x, y

    def save_as_zarr(self, zarr_path: Pathable):
        data = [FaceLandmarkSeries.load(p) for p in self.metadata_paths]
        zarr_io.save_to_zarr(data, zarr_path)
