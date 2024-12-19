from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ffmpegio
import numpy as np
from rich.progress import Progress, TaskID, TextColumn, BarColumn, TimeRemainingColumn
from visiongraph import vg

from viva.utils.path_utils import Pathable, get_files


@dataclass
class AVPreProcessingOptions:
    stream_block_size: int = 100


@dataclass
class AVPreProcessingTask:
    video_path: Path
    result_path: Path
    options: AVPreProcessingOptions


class AVPreProcessor:
    def __init__(self,
                 data_path: Pathable,
                 output_path: Pathable,
                 options: Optional[AVPreProcessingOptions] = None):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} does not exist!")

        self.output_path = Path(output_path)
        self.options = options if options is not None else AVPreProcessingOptions()
        self.videos_paths = get_files(data_path, "*.mov", "*.mp4", "*.mkv", "*.avi", recursive=True)

    def process(self, num_workers: int = 4):
        # Create tasks
        tasks = [
            AVPreProcessingTask(
                video_path=video_path.absolute(),
                result_path=self.output_path.absolute() / video_path.with_suffix(".npz").name,
                options=self.options
            )
            for video_path in self.videos_paths
        ]

        # Ensure the output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Adjust the number of workers based on the number of tasks
        num_tasks = len(tasks)
        actual_workers = min(num_workers, num_tasks)

        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                transient=False
        ) as progress:
            # Add tasks to the progress bar
            task_map = {
                progress.add_task(description=f"Processing {task.video_path.name}", total=100): task
                for task in tasks
            }

            if actual_workers <= 1:
                # Sequential execution for debugging
                for task_id, task in task_map.items():
                    self._process_task(task, progress, task_id)
            else:
                # Parallel execution with ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                    futures = [
                        executor.submit(self._process_task, task, progress, task_id)
                        for task_id, task in task_map.items()
                    ]

                    # Wait for all tasks to complete
                    for future in as_completed(futures):
                        future.result()

    @staticmethod
    def _process_task(task: AVPreProcessingTask, progress: Progress, task_id: TaskID):
        video_path = task.video_path
        result_path = task.result_path
        options = task.options

        # create face mesh estimator
        face_mesh_estimator = vg.MediaPipeFaceMeshEstimator(max_num_faces=1)
        face_mesh_estimator.setup()

        # read stream info
        video_streams = ffmpegio.probe.video_streams_basic(str(video_path))
        audio_streams = ffmpegio.probe.audio_streams_basic(str(video_path))

        video_info = video_streams[0]
        audio_infos = audio_streams[0]

        # extract video information
        video_duration_ms = float(video_info["duration"] * 1000)
        video_fps = float(video_info["frame_rate"])
        total_video_frames = int(video_duration_ms / video_fps)
        video_frame_length_ms = 1000 / video_fps
        video_width = int(video_info["width"])
        video_height = int(video_info["height"])

        # extract audio information
        audio_duration_ms = float(audio_infos["duration"] * 1000)

        # setup progressbar
        progress.update(task_id, total=total_video_frames)

        # results
        samples = []

        # read video and audio as stream
        frame_index = 0
        with ffmpegio.open(str(video_path), "rva", blocksize=options.stream_block_size) as fin:
            for frames in fin:
                video_frames: np.ndarray = frames["v:0"]
                audio_frames: np.ndarray = frames["a:0"]

                # analyze audio block

                # analyze video block
                for frame in video_frames:
                    results = face_mesh_estimator.process(frame)

                    if len(results) > 0:
                        face_mesh = results[0]
                        landmarks = vg.vector_to_array(face_mesh.landmarks)

                    progress.update(task_id, advance=1)
                    frame_index += 1

        # todo: store face samples efficiently
        progress.update(task_id, completed=total_video_frames)
        return result_path
