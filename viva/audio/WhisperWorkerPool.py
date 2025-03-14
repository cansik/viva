import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from viva.audio import whisper
from viva.worker.BaseWorker import BaseWorker
from viva.worker.BaseWorkerPool import BaseWorkerPool

logger = logging.getLogger(__name__)


@dataclass
class WhisperTask:
    audio_data: np.ndarray


class WhisperWorker(BaseWorker[WhisperTask, dict]):
    """
    A worker that loads a Whisper model and processes audio tasks.
    """

    def __init__(self, worker_id: int, task_queue_size: int = 0, model_name: str = "base"):
        super().__init__(worker_id, task_queue_size)
        self.model_name = model_name
        self.model: Optional[whisper.BaseWhisper] = None

    def setup(self):
        self.model = whisper.create_whisper()

    def handle_task(self, task: WhisperTask) -> dict:
        return self.model.transcribe(task.audio_data, word_timestamps=True)

    def cleanup(self):
        self.model = None

    def process_audio(self, audio_data: np.ndarray) -> dict:
        """
        Convenience method to submit an audio task and wait for the result.
        """
        task = WhisperTask(audio_data)
        future = self.submit_task(task)
        return future.result()


class WhisperWorkerPool(BaseWorkerPool[WhisperWorker]):
    """
    A worker pool dedicated to WhisperWorkers.
    """

    def __init__(self, num_workers: int, model_name: str = "base"):
        # Pass the model_name to each WhisperWorker via worker_kwargs.
        super().__init__(num_workers, worker_class=WhisperWorker, model_name=model_name)
