import logging
from multiprocessing import Process, Queue, Event
from typing import Optional, TypeVar, Generic, List, Callable

logger = logging.getLogger(__name__)

TTask = TypeVar('TTask')
TResult = TypeVar('TResult')


class BaseWorker(Process, Generic[TTask, TResult]):
    """
    A generic worker process that uses a task queue to receive work and a results queue to return results.
    Subclasses must implement the handle_task() method.
    """

    def __init__(self, worker_id: int, task_queue_size: int = 0):
        super().__init__(target=self._run_loop)
        self.worker_id = worker_id
        self.stop_requested = Event()
        self.tasks: Queue[Optional[TTask]] = Queue(maxsize=task_queue_size)
        self.results: Queue[TResult] = Queue(maxsize=task_queue_size)

    def _run_loop(self):
        while not self.stop_requested.is_set():
            task = self.tasks.get()
            if task is None:
                logger.debug(f"Worker {self.worker_id}: Stop signal received.")
                self.stop_requested.set()
                continue
            try:
                result = self.handle_task(task)
                self.results.put(result)
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error handling task: {e}")
        self.cleanup()

    def handle_task(self, task: TTask) -> TResult:
        """
        Process a single task. Subclasses must override this method.
        This method is called in sub-process environment.
        """
        raise NotImplementedError("Subclasses must implement handle_task.")

    def submit_task(self, task: TTask) -> TResult:
        """
        Submits a task for processing. This is called by the user.
        """
        # todo: it would be better to return a future or something like that.
        #  which later can read the processed result
        self.tasks.put(task)
        return self.results.get()

    def cleanup(self):
        """
        Optional cleanup code once the worker stops.
        """
        pass

    def stop(self):
        """
        Signals the worker to stop processing tasks.
        """
        self.tasks.put(None)


class BaseWorkerPool(Generic[TTask, TResult]):
    """
    A generic worker pool that manages multiple workers.
    The pool is initialized with a worker_class and any additional keyword arguments to pass to each worker.
    """

    def __init__(self, num_workers: int, worker_class: Callable[..., BaseWorker[TTask, TResult]], **worker_kwargs):
        self.num_workers = num_workers
        self.worker_class = worker_class
        self.worker_kwargs = worker_kwargs
        self.workers: List[BaseWorker[TTask, TResult]] = []
        self.active_worker_ids: Queue[int] = Queue()

    def start(self):
        logger.debug(f"Pool: Starting {self.num_workers} workers.")
        for i in range(self.num_workers):
            worker = self.worker_class(i, **self.worker_kwargs)
            worker.start()
            self.workers.append(worker)
            self.active_worker_ids.put(i)
        logger.debug("Pool: All workers started.")

    def acquire(self) -> BaseWorker[TTask, TResult]:
        """
        Acquire a worker from the pool (blocking until one is available).
        """
        logger.debug("Pool: Acquiring a worker.")
        worker_id = self.active_worker_ids.get()
        worker = self.workers[worker_id]
        logger.debug(f"Pool: Worker {worker.worker_id} acquired.")
        return worker

    def release(self, worker: BaseWorker[TTask, TResult]):
        """
        Release a worker back to the pool.
        """
        logger.debug(f"Pool: Releasing worker {worker.worker_id}.")
        self.active_worker_ids.put(worker.worker_id)

    def stop(self):
        """
        Stop all workers in the pool.
        """
        logger.debug("Pool: Stopping all workers.")
        for worker in self.workers:
            worker.stop()
        logger.debug("Pool: All workers stopped.")
