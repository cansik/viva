from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.utils.RangeMap import RangeMap, RangeResult
from viva.utils.path_utils import Pathable, get_files


class FaceLandmarkDataset(Dataset):

    def __init__(self, data_path: Optional[Pathable] = None, block_length: int = 15):
        super().__init__()
        self.block_length = block_length

        self.data_path: Optional[Path] = Path(data_path) if data_path is not None else None
        self.metadata_paths: List[Path] = self._load_metadata_files() if data_path is not None else []

        # create index for quick query lookup
        self.data_index = RangeMap[Path]()
        self.data_count = 0
        self.create_data_index()

    def create_data_index(self):
        self.data_index.clear()
        self.data_count = 0

        current_index = 0

        paths_to_remove = []
        for metadata_path in self.metadata_paths:
            series = FaceLandmarkSeries.load(metadata_path, metadata_only=True)

            if series is None:
                paths_to_remove.append(metadata_path)
                continue

            # todo: what if block size is larger than actual samples?!
            max_index = current_index + max(series.sample_count - self.block_length, 1)
            self.data_index.add_range(current_index, max_index, metadata_path)
            current_index = max_index
        self.data_count = current_index

        for path in paths_to_remove:
            self.metadata_paths.remove(path)

    def _load_metadata_files(self) -> List[Path]:
        if not self.data_path.is_dir():
            raise Exception("Datapath has to be a directory!")

        return get_files(self.data_path, "*.json", recursive=True)

    def get_series(self, index: int) -> Tuple[FaceLandmarkSeries, RangeResult]:
        range_result = self.data_index[index]
        metadata_path = Path(range_result.value)

        return FaceLandmarkSeries.load(metadata_path.with_suffix(".npz")), range_result

    def __len__(self) -> int:
        return self.data_count

    def __getitem__(self, index: int):
        series, range_result = self.get_series(index)

        start_index = index - range_result.start
        end_index = start_index + self.block_length

        x = series.samples[start_index:end_index]
        y = series.speaking_labels[start_index:end_index].astype(np.float32)

        return x, y

    @staticmethod
    def from_list(paths: List[Path], block_length: int = 15) -> "FaceLandmarkDataset":
        dataset = FaceLandmarkDataset(block_length=block_length)
        dataset.metadata_paths = paths
        dataset.create_data_index()
        return dataset
