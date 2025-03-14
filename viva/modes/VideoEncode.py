import argparse
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import ffmpegio
from rich.console import Console
from rich.progress import track

from viva.modes.VivaBaseMode import VivaBaseMode
from viva.utils.path_utils import get_files


@dataclass
class VideoEncodeTask:
    video_path: Path
    output_path: Path
    chunk_length: float


class VideoEncodeMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

    def run(self):
        args = self._parse_args()
        data_path = Path(args.data)
        output_path = data_path if args.output is None else Path(args.output)
        chunk_length = float(args.chunk_length)

        videos_paths = get_files(data_path, "*.mov", "*.mp4", "*.mkv", "*.avi", recursive=False)

        tasks = [VideoEncodeTask(p, output_path, chunk_length) for p in videos_paths]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for task in tasks:
                future = pool.submit(self.run_encode_task, task)
                futures.append(future)

            for future in track(futures, description="total"):
                future.result()

        self.console.print("done!")

    def run_encode_task(self, task: VideoEncodeTask):
        self.split_video(task.video_path, task.output_path, task.chunk_length)

    @staticmethod
    def split_video(input_path: Path, output_path: Path, chunk_duration: float):
        # Probe the input video to get its total duration
        video_streams = ffmpegio.probe.video_streams_basic(str(input_path))
        video_info = video_streams[0]

        # extract video information
        total_duration = float(video_info["duration"])

        # Calculate the number of chunks needed
        num_chunks = int(total_duration // chunk_duration) + (1 if total_duration % chunk_duration > 0 else 0)

        for i in range(num_chunks):
            start_time = i * chunk_duration
            output_file = output_path / f"{input_path.stem}-{i:03d}.mp4"

            ffmpegio.transcode(str(input_path), str(output_file),
                               ss=start_time, t=chunk_duration,
                               acodec="copy", vcodec='libx264', preset='slow', crf=22,
                               show_log=False)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="viva preprocess")
        parser.add_argument("data", type=str, help="Path to the videos to divide.")
        parser.add_argument("--output", default=None, type=str, help="Output path, by default data-path.")
        parser.add_argument("--chunk-length", default=10.0, type=float, help="Chunk length in s.")
        return parser.parse_args()
