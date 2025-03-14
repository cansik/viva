import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from viva.audio.SileroVAD import VADResult, SileroVAD
from viva.worker.BaseWorker import BaseWorker
from viva.worker.BaseWorkerPool import BaseWorkerPool

logger = logging.getLogger(__name__)


@dataclass
class SileroVADTask:
    audio_data: np.ndarray


class SileroVADWorker(BaseWorker[SileroVADTask, List[VADResult]]):
    def __init__(self, worker_id: int, task_queue_size: int = 0, model_name: str = "base"):
        super().__init__(worker_id, task_queue_size)
        self.model_name = model_name
        self.model: Optional[SileroVAD] = None

    def setup(self):
        self.model = SileroVAD()

    def handle_task(self, task: SileroVADTask) -> List[VADResult]:
        self.model.reset_states()
        return self.model.process(task.audio_data)

    def cleanup(self):
        self.model = None

    def process_audio(self, audio_data: np.ndarray) -> List[VADResult]:
        task = SileroVADTask(audio_data)
        future = self.submit_task(task)
        return future.result()


class SileroVADWorkerPool(BaseWorkerPool[SileroVADWorker]):
    def __init__(self, num_workers: int):
        super().__init__(num_workers, worker_class=SileroVADWorker)
