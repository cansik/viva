import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
from rich.console import Console

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.modes.VivaBaseMode import VivaBaseMode
from viva.utils.path_utils import path_serializer, get_files


class DatasetMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

    def run(self):
        args = self._parse_args()
        dataset_path = Path(args.dataset)
        output_path = dataset_path if args.output is None else Path(args.output)
        is_split = bool(args.split)

        if is_split:
            with self.console.status("splitting dataset"):
                self._split_dataset(dataset_path, output_path, args)

    def _split_dataset(self, dataset_path: Path, output_path: Path, args: argparse.Namespace):
        seed = int(args.seed)
        test_split_factor = float(args.test_split)
        val_split_factor = float(args.val_split)
        is_balance = bool(args.balance)

        # store split in dataset full dataset file
        if output_path.is_dir():
            output_path = output_path / "dataset.json"

        if not dataset_path.is_dir():
            raise Exception("Datapath has to be a directory!")

        metadata_paths = get_files(dataset_path, "*.json", recursive=True)

        # normalize metadata-paths to output path
        parent = output_path.parent
        # metadata_paths = [p.relative_to(parent) for p in metadata_paths]

        # random sampling
        random.seed(seed)
        random.shuffle(metadata_paths)

        test_count = round(len(metadata_paths) * test_split_factor)
        valid_count = round(len(metadata_paths) * val_split_factor)

        dataset = {
            "seed": seed,
            "train": metadata_paths[test_count + valid_count:],
            "test": metadata_paths[valid_count: valid_count + test_count],
            "val": metadata_paths[:valid_count]
        }

        # equalize
        if is_balance:
            for key in ("train", "val", "test"):
                dataset[key] = self._balance_samples(dataset[key])

        # test dataset split
        assert len(set(dataset["train"]).intersection(set(dataset["test"]))) == 0
        assert len(set(dataset["test"]).intersection(set(dataset["val"]))) == 0
        assert len(set(dataset["val"]).intersection(set(dataset["train"]))) == 0

        # add counts
        dataset["count"] = {
            "train": len(dataset["train"]),
            "test": len(dataset["test"]),
            "val": len(dataset["val"])
        }

        # write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(dataset, default=path_serializer, indent=2), encoding="utf-8")

    def _balance_samples(self, metadata_paths: List[Path]) -> List[Path]:
        # Load dataset
        dataset = FaceLandmarkDataset(metadata_paths=metadata_paths)

        # Separate speaking and non-speaking samples
        speaking_samples = []
        non_speaking_samples = []

        for series in dataset.data:
            if series.speaking_labels is not None:
                # check if series is more speaking or non-speaking
                is_speaking = np.sum(series.speaking_labels) > len(series.speaking_labels) / 2

                if is_speaking:
                    speaking_samples.append((series.metadata_path, series.sample_count))
                else:
                    non_speaking_samples.append((series.metadata_path, series.sample_count))

        # Calculate the target count for each category
        speaking_samples_count = sum(s[1] for s in speaking_samples)
        non_speaking_samples_count = sum(s[1] for s in non_speaking_samples)
        target_count = min(speaking_samples_count, non_speaking_samples_count)

        selected_speaking = []
        selected_non_speaking = []
        speaking_count, non_speaking_count = 0, 0

        for source, count in speaking_samples:
            if speaking_count + count <= target_count:
                selected_speaking.append(source)
                speaking_count += count

        for source, count in non_speaking_samples:
            if non_speaking_count + count <= target_count:
                selected_non_speaking.append(source)
                non_speaking_count += count

        # Combine selected paths and return
        balanced_paths = list(set(selected_speaking + selected_non_speaking))
        return balanced_paths

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="viva dataset")
        parser.add_argument("dataset", type=str, help="Dataset path.")
        parser.add_argument("--output", default=None, type=str, help="Output path, by default dataset-path.")
        parser.add_argument("--split", action="store_true", help="Split dataset into train / val / test.")
        parser.add_argument("--seed", type=int, default=12345, help="Seed for dataset creation.")
        parser.add_argument("--test-split", type=float, default=0.1, help="How many images will be used for test set.")
        parser.add_argument("--val-split", type=float, default=0.1, help="How many images will be used for valid set.")
        parser.add_argument("--balance", action="store_true",
                            help="Balances the dataset to have ~same amount of samples.")
        return parser.parse_args()
