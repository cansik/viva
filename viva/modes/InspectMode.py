import argparse
import json
from pathlib import Path

import cv2
import ffmpegio
import numpy as np
from rich.console import Console

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.modes.VivaBaseMode import VivaBaseMode
from viva.vision.vision_utils import resize_image_to_fit, annotate_landmarks


class InspectMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

    def run(self):
        args = self._parse_args()
        dataset_path = Path(args.dataset)
        dataset_mode = str(args.mode)
        block_size = int(args.block_size)

        # load datasets
        data = json.loads(dataset_path.read_text(encoding="utf-8"))

        dataset = FaceLandmarkDataset.from_list(data[dataset_mode], block_size)

        for video_path in dataset.metadata_paths:
            series = FaceLandmarkSeries.load(video_path)
            self.preview_video(series)

    def preview_video(self, series: FaceLandmarkSeries):
        video_path = Path(series.source)
        fs, frames = ffmpegio.video.read(str(video_path))

        for sample_index, frame_index in enumerate(series.video_frame_indices):
            frame_rgb = frames[frame_index]
            frame: np.ndarray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame = resize_image_to_fit(frame, 512, 512)

            frame = annotate_landmarks(frame, series.samples[sample_index])

            cv2.imshow("Inspect", frame)
            cv2.waitKey(1)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="viva inspect")
        parser.add_argument("dataset", type=str, help="Path to the dataset file.")
        parser.add_argument("--mode", default="train", type=str, help="Which mode to select.")
        parser.add_argument("--block-size", type=int, default=15,
                            help="Dataset block-size (how much data per inference block).")
        return parser.parse_args()
