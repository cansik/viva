from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import Dataset

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.utils.RangeMap import RangeMap
from viva.utils.path_utils import Pathable, get_files


class FaceLandmarkDataset(Dataset):

    def __init__(self, data_path: Pathable, block_length: int = 15):
        super().__init__()
        self.block_length = block_length

        self.data_path = Path(data_path)
        self.metadata_paths = self._load_metadata_files()

        # create index for quick query lookup
        self.data_index = RangeMap[Path]()
        self.data_count = 0
        self._create_data_index()

    def _create_data_index(self):
        self.data_index.clear()
        current_index = 0
        for metadata_path in self.metadata_paths:
            series = FaceLandmarkSeries.load(metadata_path, metadata_only=True)
            max_index = current_index + series.sample_count - self.block_length
            self.data_index.add_range(current_index, max_index, metadata_path)
            current_index = max_index
        self.data_count = current_index

    def _load_metadata_files(self) -> List[Path]:
        return get_files(self.data_path, "*.json", recursive=True)

    def __len__(self) -> int:
        return self.data_count

    def __getitem__(self, index: int):
        range_result = self.data_index[index]
        metadata_path = range_result.value

        series = FaceLandmarkSeries.load(metadata_path.with_suffix(".npz"))

        start_index = index - range_result.start
        end_index = start_index + self.block_length

        x = series.samples[start_index:end_index]
        y = series.speaking_labels[start_index:end_index].astype(np.float32)

        return x, y
