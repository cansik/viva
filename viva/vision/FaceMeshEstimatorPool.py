import logging
from dataclasses import dataclass
from multiprocessing import Queue, Process, Event
from typing import List, Optional

import numpy as np
from visiongraph import vg

logger = logging.getLogger(__name__)


@dataclass
class FaceMeshWorkOptions:
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class FaceMeshTask:
    data: np.ndarray


class FaceMeshWorker(Process):
    def __init__(self, worker_id: int,
                 task_queue_size: int = 0,
                 options: Optional[FaceMeshWorkOptions] = None
                 ):
        super().__init__(target=self._run_loop)
        self.worker_id = worker_id
        self.stop_requested = Event()
        self.face_mesh_estimator: Optional[vg.MediaPipeFaceMeshEstimator] = None
        self.tasks: Queue[Optional[FaceMeshTask]] = Queue(maxsize=task_queue_size)
        self.results: Queue[vg.ResultList[vg.BlazeFaceMesh]] = Queue(maxsize=task_queue_size)

        self.options = options if options is not None else FaceMeshWorkOptions()

        # this is used to generate a monotonic time
        self.frame_ts_id = 0
        self.frame_fps = 30

        # prepare face mesh estimator
        estimator = vg.MediaPipeFaceMeshEstimator()
        estimator.task.prepare()

    def _run_loop(self):
        logger.debug(f"Worker {self.worker_id}: Starting run loop.")
        # Start face-mesh estimator
        self.face_mesh_estimator = vg.MediaPipeFaceMeshEstimator(
            max_num_faces=1,
            min_face_detection_confidence=self.options.min_face_detection_confidence,
            min_face_presence_confidence=self.options.min_face_presence_confidence,
            min_tracking_confidence=self.options.min_tracking_confidence,
            output_facial_transformation_matrixes=True
        )
        self.face_mesh_estimator.setup()
        logger.debug(f"Worker {self.worker_id}: FaceMeshEstimator initialized.")

        while not self.stop_requested.is_set():
            task = self.tasks.get()

            if task is None:
                logger.debug(f"Worker {self.worker_id}: Stop signal received.")
                self.stop_requested.set()
                continue

            logger.debug(f"Worker {self.worker_id}: Processing frame.")
            timestamp_ms = int(self.frame_ts_id * (1000 / self.frame_fps))
            result = self.face_mesh_estimator.process(task.data, timestamp_ms=timestamp_ms)
            self.results.put(result)
            self.frame_ts_id += 1
            logger.debug(f"Worker {self.worker_id}: Frame processed successfully.")

        logger.debug(f"Worker {self.worker_id}: Releasing resources.")
        self.face_mesh_estimator.release()

    def process_frame(self, frame: np.ndarray) -> vg.ResultList[vg.BlazeFaceMesh]:
        logger.debug(f"Worker {self.worker_id}: Submitting frame for processing.")
        task = FaceMeshTask(frame)
        self.tasks.put(task)
        return self.results.get()

    def reset(self):
        logger.debug(f"Worker {self.worker_id}: Resetting FaceMeshEstimator.")
        task = FaceMeshTask(np.zeros((100, 100, 3), dtype=np.uint8))
        self.tasks.put(task)
        self.results.get()

    def stop(self):
        logger.debug(f"Worker {self.worker_id}: Sending stop signal.")
        self.tasks.put(None)


class FaceMeshEstimatorPool:
    def __init__(self, num_workers: int, worker_options: Optional[FaceMeshWorkOptions] = None):
        self.num_workers = num_workers
        self.worker_options = worker_options if worker_options is not None else FaceMeshWorkOptions()
        self.active_workers: Queue[int] = Queue()
        self.workers: List[FaceMeshWorker] = []

    def start(self):
        logger.debug(f"Pool: Starting {self.num_workers} workers.")
        for i in range(self.num_workers):
            worker = FaceMeshWorker(i, options=self.worker_options)
            worker.start()

            self.workers.append(worker)
            self.active_workers.put(worker.worker_id)
        logger.debug("Pool: All workers started.")

    def acquire(self) -> FaceMeshWorker:
        logger.debug("Pool: Acquiring a worker.")
        worker_id = self.active_workers.get()
        worker = self.workers[worker_id]
        logger.debug(f"Pool: Worker {worker.worker_id} acquired.")
        return worker

    def release(self, worker: FaceMeshWorker):
        logger.debug(f"Pool: Releasing worker {worker.worker_id}.")
        self.active_workers.put(worker.worker_id)

    def stop(self):
        logger.debug("Pool: Stopping all workers.")
        for worker in self.workers:
            worker.stop()
        logger.debug("Pool: All workers stopped.")


def main():
    pool = FaceMeshEstimatorPool(1)
    pool.start()

    worker = pool.acquire()
    for i in range(0, 10):
        result = worker.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        print(f"Results: {len(result)}")

    pool.release(worker)
    pool.stop()


if __name__ == "__main__":
    main()
