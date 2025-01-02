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
                 transforms: Optional[List[BaseLandmarkAugmentation]] = None,
                 augmentations: Optional[List[BaseLandmarkAugmentation]] = None):
        super().__init__()
        self.block_length = block_length

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
        indices_to_remove = set()
        for i, (series, metadata_path) in enumerate(list(zip(all_series, self.metadata_paths))):
            if series is None or series.sample_count < self.block_length:
                indices_to_remove.add(i)
                continue

            # todo: what if block size is larger than actual samples?!
            max_index = current_index + max(series.sample_count - self.block_length, 1)
            self.data_index.add_range(current_index, max_index, i)
            current_index = max_index

        # remove all indices
        self.metadata_paths = [item for i, item in enumerate(self.metadata_paths) if i not in indices_to_remove]
        all_series = [item for i, item in enumerate(all_series) if i not in indices_to_remove]

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
        end_index = start_index + self.block_length

        x = series.samples[start_index:end_index]
        y = series.speaking_labels[start_index:end_index].astype(np.float32)

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
