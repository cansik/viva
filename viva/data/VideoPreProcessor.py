import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import cv2
import ffmpegio
import numpy as np
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from visiongraph import vg

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.utils.path_utils import Pathable, get_files
from viva.vision.FaceMeshEstimatorPool import FaceMeshEstimatorPool


@dataclass
class VideoPreProcessingOptions:
    stream_block_size: int = 100
    is_speaking: bool = True
    is_debug: bool = False


@dataclass
class VideoPreProcessingTask:
    video_path: Path
    result_path: Path
    options: VideoPreProcessingOptions


class VideoPreProcessor:
    def __init__(self,
                 data_path: Pathable,
                 output_path: Pathable,
                 options: Optional[VideoPreProcessingOptions] = None,
                 num_workers: int = 4):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} does not exist!")

        self.num_workers = num_workers
        self.output_path = Path(output_path)
        self.options = options if options is not None else VideoPreProcessingOptions()
        self.videos_paths = get_files(data_path, "*.mov", "*.mp4", "*.mkv", "*.avi", recursive=True)

        # create face mesh estimator pool
        self.face_mesh_pool = FaceMeshEstimatorPool(self.num_workers)

    def process(self):
        # startup pol
        self.face_mesh_pool.start()

        # Create tasks
        tasks = [
            VideoPreProcessingTask(
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
        actual_workers = min(self.num_workers, num_tasks)

        start_time = time.time()
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                MofNCompleteColumn()
        ) as progress:
            # Create an overall progress bar
            overall_task_id = progress.add_task(description="Overall Progress", total=num_tasks)

            # Sequential execution for debugging or single worker
            if actual_workers <= 1:
                for task in tasks:
                    self._process_task(task, progress)
                    progress.advance(overall_task_id)
            else:
                # Parallel execution with ProcessPoolExecutor
                with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    futures = {}
                    for task in tasks:
                        # Submit tasks to executor and track their futures
                        future = executor.submit(self._process_task, task, progress)
                        futures[future] = task

                    # Process tasks as workers become available
                    for future in as_completed(futures):
                        # Wait for the future to complete
                        future.result()
                        progress.advance(overall_task_id)

        self.face_mesh_pool.stop()
        end_time = time.time()
        print(f"It took {str(timedelta(seconds=end_time - start_time))} seconds to process.")

    def _process_task(self, task: VideoPreProcessingTask, progress: Progress):
        task_id = progress.add_task(description=f"Processing {task.video_path.name}", total=100)

        video_path = task.video_path
        result_path = task.result_path
        options = task.options

        # read stream info
        video_streams = ffmpegio.probe.video_streams_basic(str(video_path))
        video_info = video_streams[0]

        # extract video information
        video_duration_ms = float(video_info["duration"] * 1000)
        video_fps = float(video_info["frame_rate"])
        total_video_frames = int(video_duration_ms / video_fps)
        video_frame_length_ms = 1000 / video_fps
        video_width = int(video_info["width"])
        video_height = int(video_info["height"])

        # setup progressbar
        progress.update(task_id, total=total_video_frames)

        video_frame_indices = []
        samples = []
        transforms = []
        is_speaking_labels = []

        face_mesh_estimator = self.face_mesh_pool.acquire()
        face_mesh_estimator.reset()

        # read video and audio as stream
        frame_index = 0
        with ffmpegio.open(str(video_path), "rv", blocksize=options.stream_block_size) as fin:
            for frames in fin:
                video_frames: np.ndarray = frames

                # analyze video block
                for frame_rgb in video_frames:
                    # todo: loading directly in BGR format
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    results = face_mesh_estimator.process_frame(frame)

                    if len(results) > 0:
                        face_mesh = results[0]
                        landmarks = vg.vector_to_array(face_mesh.landmarks)
                        transform = face_mesh.transformation_matrix

                        # add data to lists
                        video_frame_indices.append(frame_index)
                        samples.append(landmarks)
                        transforms.append(transform)
                        is_speaking_labels.append(options.is_speaking)

                        if options.is_debug:
                            preview = frame.copy()
                            face_mesh.annotate(preview)
                            cv2.imshow("Landmarks", preview)
                            cv2.waitKey(0)

                    progress.update(task_id, advance=1)
                    frame_index += 1

        self.face_mesh_pool.release(face_mesh_estimator)

        # create and store series
        landmark_series = FaceLandmarkSeries(str(video_path), video_width, video_height, video_fps, len(samples),
                                             np.array(video_frame_indices, dtype=np.uint32),
                                             np.array(samples, dtype=np.float32),
                                             np.array(transforms, dtype=np.float32),
                                             np.array(is_speaking_labels, dtype=bool))
        landmark_series.save(result_path)

        progress.update(task_id, completed=total_video_frames)
        progress.remove_task(task_id)
        return result_path
