import logging
from dataclasses import dataclass
from multiprocessing import Queue, Process, Event
from typing import List, Optional

import numpy as np

from viva.audio import whisper

logger = logging.getLogger(__name__)


@dataclass
class WhisperTask:
    audio_data: np.ndarray


class WhisperWorker(Process):
    def __init__(self, worker_id: int, task_queue_size: int = 0, model_name: str = "base"):
        super().__init__(target=self._run_loop)
        self.worker_id = worker_id
        self.stop_requested = Event()
        self.tasks: Queue[Optional[WhisperTask]] = Queue(maxsize=task_queue_size)
        self.results: Queue[dict] = Queue(maxsize=task_queue_size)
        self.model_name = model_name
        self.model: Optional[whisper.BaseWhisper] = None

    def _run_loop(self):
        logger.debug(f"Worker {self.worker_id}: Starting run loop.")
        # Load the Whisper model
        self.model = whisper.create_whisper()
        logger.debug(f"Worker {self.worker_id}: Whisper model '{self.model_name}' loaded.")

        while not self.stop_requested.is_set():
            task = self.tasks.get()

            if task is None:
                logger.debug(f"Worker {self.worker_id}: Stop signal received.")
                self.stop_requested.set()
                continue

            logger.debug(f"Worker {self.worker_id}: Processing audio data.")
            try:
                # Perform transcription
                result = self.model.transcribe(task.audio_data, word_timestamps=True)
                self.results.put(result)
                logger.debug(f"Worker {self.worker_id}: Audio processed successfully.")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error processing audio - {e}")

        logger.debug(f"Worker {self.worker_id}: Releasing resources.")
        self.model = None

    def process_audio(self, audio_data: np.ndarray) -> dict:
        logger.debug(f"Worker {self.worker_id}: Submitting audio for processing.")
        task = WhisperTask(audio_data)
        self.tasks.put(task)
        return self.results.get()

    def stop(self):
        logger.debug(f"Worker {self.worker_id}: Sending stop signal.")
        self.tasks.put(None)


class WhisperWorkerPool:
    def __init__(self, num_workers: int, model_name: str = "base"):
        self.num_workers = num_workers
        self.model_name = model_name
        self.active_workers: Queue[int] = Queue()
        self.workers: List[WhisperWorker] = []

    def start(self):
        logger.debug(f"Pool: Starting {self.num_workers} workers.")
        for i in range(self.num_workers):
            worker = WhisperWorker(i, model_name=self.model_name)
            worker.start()

            self.workers.append(worker)
            self.active_workers.put(worker.worker_id)
        logger.debug("Pool: All workers started.")

    def acquire(self) -> WhisperWorker:
        logger.debug("Pool: Acquiring a worker.")
        worker_id = self.active_workers.get()
        worker = self.workers[worker_id]
        logger.debug(f"Pool: Worker {worker.worker_id} acquired.")
        return worker

    def release(self, worker: WhisperWorker):
        logger.debug(f"Pool: Releasing worker {worker.worker_id}.")
        self.active_workers.put(worker.worker_id)

    def stop(self):
        logger.debug("Pool: Stopping all workers.")
        for worker in self.workers:
            worker.stop()
        logger.debug("Pool: All workers stopped.")


def main():
    pool = WhisperWorkerPool(num_workers=1, model_name="base")
    pool.start()

    worker = pool.acquire()
    # Simulate an audio numpy array (replace with actual audio data)
    fake_audio = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds of fake audio
    result = worker.process_audio(fake_audio)
    print(f"Transcription: {result['text']}")

    pool.release(worker)
    pool.stop()


if __name__ == "__main__":
    main()
