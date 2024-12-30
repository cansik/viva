import argparse
import json
from pathlib import Path

import cv2
import ffmpegio
import numpy as np
from rich.console import Console
from tqdm import tqdm

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.NormalizeLandmarks import NormalizeLandmarks
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
        display_normalized_landmarks = bool(args.norm)

        transforms = []
        if display_normalized_landmarks:
            transforms.append(NormalizeLandmarks())

        # load dataset
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset = FaceLandmarkDataset(metadata_paths=data[dataset_mode], block_length=block_size, transforms=transforms)

        if display_normalized_landmarks:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            h, w = image.shape[:2]
            for series in tqdm(dataset.dataset, desc="analyzing", total=len(dataset)):
                samples = series.samples
                samples[:, :, 0] = ((samples[:, :, 0] + 1) / 2) * w
                samples[:, :, 1] = ((samples[:, :, 1] + 1) / 2) * h

                for sample in samples:
                    image.fill(0)
                    for lm in sample:
                        center = round(lm[0]), round(lm[1])
                        cv2.circle(image, center, 1, (0, 0, 255), -1)
                    cv2.imshow("Normalized Landmarks", image)
                    cv2.waitKey(0)

        speaking_count = 0
        not_speaking_count = 0
        for series in tqdm(dataset.dataset, desc="analyzing", total=len(dataset)):
            speaking_labels = series.speaking_labels.flatten()

            speaking_count += speaking_labels.sum()
            not_speaking_count += len(speaking_labels) - speaking_labels.sum()

        self.console.print(f"Speaking:     {speaking_count}")
        self.console.print(f"Not Speaking: {not_speaking_count}")

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
        parser.add_argument("--norm", action="store_true", help="Display normalized landmarks.")
        return parser.parse_args()
