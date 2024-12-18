import argparse
from pathlib import Path

from rich.console import Console

from viva.data.VideoPreProcessor import VideoPreProcessor, VideoPreProcessingOptions
from viva.modes.VivaBaseMode import VivaBaseMode


class PreProcessMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

    def run(self):
        args = self._parse_args()
        data_path = Path(args.data)
        output_path = data_path if args.output is None else Path(args.output)
        is_speaking = bool(args.speaking)
        is_debug = bool(args.debug)
        num_workers = int(args.num_workers)
        stream_block_size = int(args.block_size)

        options = VideoPreProcessingOptions(stream_block_size=stream_block_size,
                                            is_speaking=is_speaking,
                                            is_debug=is_debug)

        p = VideoPreProcessor(data_path, output_path, options)
        p.process(num_workers=num_workers)

        self.console.print("done!")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="viva preprocess")
        parser.add_argument("data", type=str, help="Path to the data to pre-process.")
        parser.add_argument("--output", default=None, type=str, help="Output path, by default data-path.")
        parser.add_argument("--speaking", action="store_true", help="If the data is speaking data.")
        parser.add_argument("--debug", action="store_true", help="If is in debug mode.")
        parser.add_argument("--num-workers", default=4, type=int, help="Num parallel worker threads.")
        parser.add_argument("--block-size", default=100, type=int, help="Stream block size.")
        return parser.parse_args()
