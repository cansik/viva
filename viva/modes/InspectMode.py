import argparse
import json
from pathlib import Path

import cv2
import ffmpegio
import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from visiongraph import vg

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.FilterLandmarkIndices import FilterLandmarkIndices
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
        stride = int(args.stride)
        display_normalized_landmarks = bool(args.norm)
        display_samples = bool(args.samples)

        transforms = []
        if display_normalized_landmarks:
            transforms.append(NormalizeLandmarks())

        if display_samples:
            transforms.append(FilterLandmarkIndices(vg.BlazeFaceMesh.FEATURES_148))

        # load dataset
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset = FaceLandmarkDataset(metadata_paths=data[dataset_mode],
                                      block_length=block_size,
                                      stride=stride,
                                      transforms=transforms)

        if display_samples:
            sample_image_size = 256
            image = np.zeros((sample_image_size, sample_image_size * block_size, 3), dtype=np.uint8)
            for i in tqdm(range(len(dataset)), desc="samples"):
                x, y = dataset[i]

                self.preview_samples(x, y, image)

                cv2.imshow("Samples", image)
                cv2.waitKey(1)
            exit(0)

        if display_normalized_landmarks:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            h, w = image.shape[:2]
            for series in tqdm(dataset.data, desc="analyzing", total=len(dataset)):
                samples = series.samples
                samples[:, :, 0] = ((samples[:, :, 0] + 1) / 2) * w
                samples[:, :, 1] = h - ((samples[:, :, 1] + 1) / 2) * h

                speaking_labels = series.speaking_labels
                color = (0, 255, 0) if speaking_labels[0] else (0, 0, 255)

                for sample in samples:
                    image.fill(0)
                    for lm in sample:
                        center = round(lm[0]), round(lm[1])
                        cv2.circle(image, center, 1, color, -1)
                    cv2.imshow("Normalized Landmarks", image)
                    cv2.waitKey(1)
            exit(0)

        speaking_count = 0
        not_speaking_count = 0
        counts = []

        for series in tqdm(dataset.data, desc="analyzing", total=len(dataset)):
            speaking_labels = series.speaking_labels.flatten()
            count = len(speaking_labels)

            speaking_count += speaking_labels.sum()
            not_speaking_count += count - speaking_labels.sum()

            counts.append(count)

        counts = np.array(counts)
        hist = np.histogram(counts, bins=np.unique(counts))

        self.console.print(f"Speaking:     {speaking_count}")
        self.console.print(f"Not Speaking: {not_speaking_count}")

        # print histogram
        bins = hist[1]  # Bin edges
        frequencies = hist[0]  # Bin frequencies

        # Create the console and table
        table = Table(title="Histogram")

        # Add columns to the table
        table.add_column("Bin Range", justify="center", style="cyan")
        table.add_column("Frequency", justify="center", style="magenta")

        # Fill the table with histogram data
        for i in range(len(frequencies)):
            bin_range = f"[{bins[i]:.2f}, {bins[i + 1]:.2f})"
            table.add_row(bin_range, str(frequencies[i]))

        # Print the table
        self.console.print(table)

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

    def preview_samples(self, x: np.ndarray, y: np.ndarray, image: np.ndarray):
        h, w = image.shape[:2]
        image.fill(0)

        sample_cell_size = w // len(x)

        x[:, :, 0] *= sample_cell_size
        x[:, :, 1] *= sample_cell_size

        for i, sample in enumerate(x):
            color = (0, 255, 0) if y[i] else (0, 0, 255)
            x0 = round(i * sample_cell_size)

            cv2.rectangle(image, (x0, 0), (x0 + sample_cell_size - 1, sample_cell_size - 1), (255, 255, 255), 1)

            for lm in sample:
                center = round(x0 + lm[0]), round(lm[1])
                cv2.circle(image, center, 1, color, -1)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="viva inspect")
        parser.add_argument("dataset", type=str, help="Path to the dataset file.")
        parser.add_argument("--mode", default="train", type=str, help="Which mode to select.")
        parser.add_argument("--block-size", type=int, default=15,
                            help="Dataset block-size (how much data per inference block).")
        parser.add_argument("--stride", type=int, default=1, help="Stride of the samples.")
        parser.add_argument("--samples", action="store_true", help="Display samples.")
        parser.add_argument("--norm", action="store_true", help="Display normalized landmarks.")
        return parser.parse_args()
