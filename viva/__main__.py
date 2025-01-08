import argparse
import sys
from typing import Dict, Type, Tuple, List

from rich.console import Console

from viva.modes.DatasetMode import DatasetMode
from viva.modes.DemoMode import DemoMode
from viva.modes.ExportMode import ExportMode
from viva.modes.InspectMode import InspectMode
from viva.modes.PreProcessMode import PreProcessMode
from viva.modes.TrainMode import TrainMode
from viva.modes.VivaBaseMode import VivaBaseMode

viva_modes: Dict[str, Type[VivaBaseMode]] = {
    "preprocess": PreProcessMode,
    "dataset": DatasetMode,
    "inspect": InspectMode,
    "train": TrainMode,
    "demo": DemoMode,
    "export": ExportMode,
}


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(prog="viva", add_help=False)
    parser.add_argument("mode", choices=viva_modes.keys(), help="Which mode to start viva in.")
    return parser.parse_known_args()


def main() -> None:
    args, unknown_args = parse_args()
    sys.argv = [sys.argv[0], *unknown_args]

    mode_str = str(args.mode)
    mode_type = viva_modes[mode_str]

    console = Console()
    console.print("Viva - Visual Voice Activation Detection")
    console.print(f"Mode: {mode_str}")
    mode = mode_type(console)
    mode.run()


if __name__ == "__main__":
    main()
