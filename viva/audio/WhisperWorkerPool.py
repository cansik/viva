import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from viva.audio import whisper
from viva.worker.BaseWorker import BaseWorker, BaseWorkerPool

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

    def _run_loop(self):
        logger.debug(f"Worker {self.worker_id}: Starting run loop.")
        # Load the Whisper model once when the worker starts
        self.model = whisper.create_whisper()
        logger.debug(f"Worker {self.worker_id}: Whisper model '{self.model_name}' loaded.")
        # Enter the generic loop from the base class
        super()._run_loop()

    def handle_task(self, task: WhisperTask) -> dict:
        logger.debug(f"Worker {self.worker_id}: Processing audio data.")
        result = self.model.transcribe(task.audio_data, word_timestamps=True)
        logger.debug(f"Worker {self.worker_id}: Audio processed successfully.")
        return result

    def cleanup(self):
        logger.debug(f"Worker {self.worker_id}: Releasing resources.")
        self.model = None

    def process_audio(self, audio_data: np.ndarray) -> dict:
        """
        Convenience method to submit an audio task and wait for the result.
        """
        logger.debug(f"Worker {self.worker_id}: Submitting audio for processing.")
        task = WhisperTask(audio_data)
        self.tasks.put(task)
        return self.results.get()


class WhisperWorkerPool(BaseWorkerPool[WhisperTask, dict]):
    """
    A worker pool dedicated to WhisperWorkers.
    """

    def __init__(self, num_workers: int, model_name: str = "base"):
        # Pass the model_name to each WhisperWorker via worker_kwargs.
        super().__init__(num_workers, worker_class=WhisperWorker, model_name=model_name)
