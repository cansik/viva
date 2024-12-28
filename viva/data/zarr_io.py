from dataclasses import asdict
from typing import List, Optional

import numpy as np
import zarr

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.utils.path_utils import Pathable


def save_to_zarr(series_list: List[FaceLandmarkSeries], zarr_path: Pathable) -> None:
    """
    Save a list of FaceLandmarkSeries into a single Zarr file.

    :param series_list: List of FaceLandmarkSeries to save.
    :param zarr_path: Path to the Zarr file.
    """
    # Create a Zarr group
    root = zarr.open_group(zarr_path, mode="w")

    for idx, series in enumerate(series_list):
        group = root.create_group(f"series_{idx}")

        # Store metadata
        metadata = asdict(series)
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                continue  # Skip arrays for now
            group.attrs[key] = value

        # Store arrays
        if series.video_frame_indices is not None:
            group.create_dataset("video_frame_indices", data=series.video_frame_indices, compressor=zarr.Blosc())
        if series.samples is not None:
            group.create_dataset("samples", data=series.samples, compressor=zarr.Blosc())
        if series.transforms is not None:
            group.create_dataset("transforms", data=series.transforms, compressor=zarr.Blosc())
        if series.speaking_labels is not None:
            group.create_dataset("speaking_labels", data=series.speaking_labels, compressor=zarr.Blosc())


def load_from_zarr(zarr_path: Pathable, metadata_only: bool = False) -> List[FaceLandmarkSeries]:
    """
    Load a list of FaceLandmarkSeries from a Zarr file.

    :param zarr_path: Path to the Zarr file.
    :param metadata_only: If True, only load metadata without loading numpy arrays.
    :return: List of FaceLandmarkSeries instances.
    """
    root = zarr.open_group(zarr_path, mode="r")
    series_list = []

    for group_name in root:
        group = root[group_name]
        metadata = {attr: group.attrs[attr] for attr in group.attrs.keys()}

        # Reconstruct the FaceLandmarkSeries
        series = FaceLandmarkSeries(
            source=metadata.get("source", ""),
            video_width=metadata["video_width"],
            video_height=metadata["video_height"],
            video_fps=metadata["video_fps"],
            sample_count=metadata["sample_count"],
        )

        if not metadata_only:
            # Load arrays if they exist
            if "video_frame_indices" in group:
                series.video_frame_indices = group["video_frame_indices"][:]
            if "samples" in group:
                series.samples = group["samples"][:]
            if "transforms" in group:
                series.transforms = group["transforms"][:]
            if "speaking_labels" in group:
                series.speaking_labels = group["speaking_labels"][:]

        series_list.append(series)

    return series_list


def load_from_zarr_by_index(zarr_path: Pathable,
                            index: int,
                            metadata_only: bool = False) -> Optional[FaceLandmarkSeries]:
    """
    Load a specific FaceLandmarkSeries from a Zarr file by index.

    :param zarr_path: Path to the Zarr file.
    :param index: Index of the series to load.
    :param metadata_only: If True, only load metadata without loading numpy arrays.
    :return: The specified FaceLandmarkSeries instance, or None if the index is invalid.
    """
    root = zarr.open_group(zarr_path, mode="r")
    group_name = f"series_{index}"

    if group_name not in root:
        return None

    group = root[group_name]
    metadata = {attr: group.attrs[attr] for attr in group.attrs.keys()}

    # Reconstruct the FaceLandmarkSeries
    series = FaceLandmarkSeries(
        source=metadata.get("source", ""),
        video_width=metadata["video_width"],
        video_height=metadata["video_height"],
        video_fps=metadata["video_fps"],
        sample_count=metadata["sample_count"],
    )

    if not metadata_only:
        # Load arrays if they exist
        if "video_frame_indices" in group:
            series.video_frame_indices = group["video_frame_indices"][:]
        if "samples" in group:
            series.samples = group["samples"][:]
        if "transforms" in group:
            series.transforms = group["transforms"][:]
        if "speaking_labels" in group:
            series.speaking_labels = group["speaking_labels"][:]

    return series
