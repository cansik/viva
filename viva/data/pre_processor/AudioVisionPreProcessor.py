from typing import Optional
from typing import Optional
from typing import Sequence

import ffmpegio
import numpy as np
import yaml

from viva.audio.SileroVADWorkerPool import SileroVADWorkerPool
from viva.audio.VADModels import convert_vad_results_to_segments
from viva.data.pre_processor.VideoPreProcessor import VideoPreProcessor, VideoPreProcessingOptions, \
    VideoPreProcessingTask
from viva.utils.path_utils import Pathable


class AudioVisionPreProcessor(VideoPreProcessor):
    def __init__(self,
                 data_path: Pathable,
                 output_path: Pathable,
                 options: Optional[VideoPreProcessingOptions] = None,
                 num_workers: int = 4,
                 num_face_mesh_workers: int = 4,
                 num_vad_workers: int = 1,
                 cache_vad_output: bool = True):
        super().__init__(data_path, output_path, options, num_workers, num_face_mesh_workers)

        self.vad_pool = SileroVADWorkerPool(num_vad_workers)
        self.cache_whisper_output = cache_vad_output

    def _start_processors(self):
        super()._start_processors()
        self.vad_pool.start()

    def _stop_processors(self):
        super()._stop_processors()
        self.vad_pool.stop()

    def _generate_speaking_labels(self, task: VideoPreProcessingTask,
                                  video_frame_count: int,
                                  video_duration_seconds: float) -> Sequence[bool]:
        # read audio stream for whisper
        fs, x = ffmpegio.audio.read(str(task.video_path), sample_fmt="dbl", ac=1, ar=16000)
        x = x.reshape(-1)

        # todo: enable caching vad results

        # run vad inference
        vad_worker = self.vad_pool.acquire()
        result = vad_worker.process_audio(x)
        self.vad_pool.release(vad_worker)

        # store vad output
        if self.cache_whisper_output:
            yaml_text = yaml.dump(result, default_flow_style=False)
            task.result_path.with_suffix(".yml").write_text(yaml_text, encoding="utf-8")

        # labels
        speaking_labels = np.full(video_frame_count + 1, False)
        vad_segments = convert_vad_results_to_segments(result, max_samples=len(x))

        # convert segments into video frame labels
        for segment in vad_segments:
            # extract timestamps in seconds
            start_ts = segment.start / fs
            end_ts = segment.end / fs

            start_frame_index = round(start_ts / video_duration_seconds * video_frame_count)
            end_frame_index = round(end_ts / video_duration_seconds * video_frame_count)

            if end_frame_index > video_frame_count:
                end_frame_index = video_frame_count

            speaking_labels[start_frame_index:end_frame_index] = True

        return speaking_labels
