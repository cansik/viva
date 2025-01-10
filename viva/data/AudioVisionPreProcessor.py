from typing import Optional
from typing import Sequence

import ffmpegio
import numpy as np

from viva.audio.WhisperWorkerPool import WhisperWorkerPool
from viva.data.VideoPreProcessor import VideoPreProcessor, VideoPreProcessingOptions, VideoPreProcessingTask
from viva.utils.path_utils import Pathable


class AudioVisionPreProcessor(VideoPreProcessor):
    def __init__(self,
                 data_path: Pathable,
                 output_path: Pathable,
                 options: Optional[VideoPreProcessingOptions] = None,
                 num_workers: int = 4,
                 num_face_mesh_workers: int = 4,
                 num_whisper_workers: int = 1):
        super().__init__(data_path, output_path, options, num_workers, num_face_mesh_workers)

        self.whisper_pool = WhisperWorkerPool(num_whisper_workers)

    def _start_processors(self):
        super()._start_processors()
        self.whisper_pool.start()

    def _stop_processors(self):
        super()._stop_processors()
        self.whisper_pool.stop()

    def _generate_speaking_labels(self, task: VideoPreProcessingTask,
                                  video_frame_count: int,
                                  video_duration_ms: float) -> Sequence[bool]:
        # video duration
        video_duration_seconds = video_duration_ms / 1000

        # read audio stream for whisper
        fs, x = ffmpegio.audio.read(str(task.video_path), sample_fmt="dbl", ac=1, ar=16000)
        x = x.reshape(-1)

        # run whisper inference
        whisper_worker = self.whisper_pool.acquire()
        result = whisper_worker.process_audio(x)
        self.whisper_pool.release(whisper_worker)

        # labels
        speaking_labels = np.full(video_frame_count, False)

        # convert segments into video frame labels
        for segment in result["segments"]:
            # extract timestamps in seconds
            start_ts = segment["start"]
            end_ts = segment["end"]

            start_frame_index = round(start_ts / video_duration_seconds * video_frame_count)
            end_frame_index = round(end_ts / video_duration_seconds * video_frame_count)

            if end_frame_index > video_frame_count:
                end_frame_index = video_frame_count

            speaking_labels[start_frame_index:end_frame_index] = True

        return speaking_labels
