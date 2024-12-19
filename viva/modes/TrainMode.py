import argparse
import json
from pathlib import Path

from rich.console import Console

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.modes.VivaBaseMode import VivaBaseMode


class TrainMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

    def run(self):
        args = self._parse_args()
        dataset_path = Path(args.dataset)
        block_size = int(args.block_size)

        # load datasets
        data = json.loads(dataset_path.read_text(encoding="utf-8"))

        train_dataset = FaceLandmarkDataset.from_list(data["train"], block_size)
        test_dataset = FaceLandmarkDataset.from_list(data["test"], block_size)
        val_dataset = FaceLandmarkDataset.from_list(data["val"], block_size)

        for i in range(len(test_dataset)):
            series = test_dataset.get_series(i)
            print(series)

        print("data")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="viva preprocess")
        parser.add_argument("dataset", type=str, help="Path to the dataset file.")
        parser.add_argument("--block-size", type=int, default=15,
                            help="Dataset block-size (how much data per inference block).")
        return parser.parse_args()
