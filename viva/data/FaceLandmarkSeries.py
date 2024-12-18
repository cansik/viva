import json
from dataclasses import fields, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FaceLandmarkSeries:
    source: str
    video_width: int
    video_height: int
    video_fps: float
    sample_count: int = 0

    # video frame indices (n,) int32
    video_frame_indices: Optional[np.ndarray] = None

    # samples array (n, 487, 5) float32
    samples: Optional[np.ndarray] = None

    # is speaking labels (n,) bool
    speaking_labels: Optional[np.ndarray] = None

    def save(self, path: Path):
        path = Path(path)

        # Separate metadata and numpy arrays
        metadata = {}
        arrays = {}

        for field in fields(self):
            value = getattr(self, field.name)

            if value is None:
                continue

            if isinstance(value, np.ndarray):
                arrays[field.name] = value
            elif isinstance(value, Path):
                metadata[field.name] = str(value)
            else:
                metadata[field.name] = value

        # Save metadata as a JSON file
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file)

        # Save arrays as an NPZ file
        arrays_path = path.with_suffix(".npz")
        np.savez(arrays_path, **arrays)

    @staticmethod
    def load(path: Path, metadata_only: bool = False) -> "FaceLandmarkSeries":
        path = Path(path)

        # Load metadata from the JSON file
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)

        # Initialize the object with metadata
        obj = FaceLandmarkSeries(**metadata)

        # Load arrays from the NPZ file if not metadata_only
        if not metadata_only:
            arrays_path = path.with_suffix(".npz")
            with np.load(arrays_path) as npz_file:
                for name in npz_file.files:
                    setattr(obj, name, npz_file[name])

        return obj
