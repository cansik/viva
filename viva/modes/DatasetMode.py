import argparse

from rich.console import Console

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.modes.VivaBaseMode import VivaBaseMode


class DatasetMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

    def run(self):
        args = self._parse_args()

        dataset = FaceLandmarkDataset("wildvvad/")
        first = dataset[0]

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        return parser.parse_args()
