import argparse
import random
from pathlib import Path

from rich.console import Console

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.modes.VivaBaseMode import VivaBaseMode


class DatasetMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

    def run(self):
        args = self._parse_args()
        dataset_path = Path(args.dataset)
        output_path = dataset_path if args.output is None else Path(args.output)
        is_split = bool(args.split)

        with self.console.status("loading dataset"):
            dataset = FaceLandmarkDataset(dataset_path)

        if is_split:
            with self.console.status("splitting dataset"):
                self._split_dataset(dataset, output_path, args)

    def _split_dataset(self, dataset: FaceLandmarkDataset, output_path: Path, args: argparse.Namespace):
        seed = int(args.seed)
        test_split_factor = float(args.test_split)
        val_split_factor = float(args.val_split)

        metadata_paths = dataset.metadata_paths
        random.seed(seed)
        random.shuffle(metadata_paths)

        test_count = round(len(metadata_paths) * test_split_factor)
        valid_count = round(len(metadata_paths) * val_split_factor)

        dataset = {
            "train": metadata_paths[test_count + valid_count:],
            "test": metadata_paths[valid_count: valid_count + test_count],
            "val": metadata_paths[:valid_count]
        }

        # test dataset split
        assert len(set(dataset["train"]).intersection(set(dataset["test"]))) == 0
        assert len(set(dataset["test"]).intersection(set(dataset["val"]))) == 0
        assert len(set(dataset["val"]).intersection(set(dataset["train"]))) == 0

        # store split in dataset cache files
        output_path.mkdir(parents=True, exist_ok=True)
        for name, paths in dataset.items():
            (output_path / "")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("dataset", type=str, help="Dataset path.")
        parser.add_argument("--output", default=None, type=str, help="Output path, by default dataset-path.")
        parser.add_argument("--split", action="store_true", help="Split dataset into train / val / test.")
        parser.add_argument("--seed", type=int, default=12345, help="Seed for dataset creation.")
        parser.add_argument("--test-split", type=float, default=0.2, help="How many images will be used for test set.")
        parser.add_argument("--val-split", type=float, default=0.2, help="How many images will be used for valid set.")
        return parser.parse_args()
