import logging
import threading
from multiprocessing import Process, Queue, Event, current_process
from typing import Optional, TypeVar, Generic, Dict, Callable, List

logger = logging.getLogger(__name__)

TTask = TypeVar('TTask')
TResult = TypeVar('TResult')


class Future(Generic[TResult]):
    """
    A very simple Future implementation.
    """

    def __init__(self):
        self._done = threading.Event()
        self._result: Optional[TResult] = None
        self._exception: Optional[Exception] = None

    def set_result(self, result: TResult):
        self._result = result
        self._done.set()

    def set_exception(self, exception: Exception):
        self._exception = exception
        self._done.set()

    def result(self, timeout: Optional[float] = None) -> TResult:
        if self._done.wait(timeout):
            if self._exception:
                raise self._exception
            return self._result  # type: ignore
        else:
            raise TimeoutError("Future result not available within timeout.")

    def done(self) -> bool:
        return self._done.is_set()


class BaseWorker(Process, Generic[TTask, TResult]):
    """
    A generic worker process that uses a task queue to receive work and a results queue
    to return results. Task submission returns a Future, which is resolved when the
    worker places the result on the results queue.

    Subclasses must implement the handle_task() method.
    """

    def __init__(self, worker_id: int, task_queue_size: int = 0):
        super().__init__(target=self._run_loop)
        self.worker_id = worker_id
        self.stop_requested = Event()
        # The tasks queue will carry tuples: (task_id, TTask)
        self.tasks: Queue = Queue(maxsize=task_queue_size)
        # The results queue will carry tuples: (task_id, TResult or Exception)
        self.results: Queue = Queue(maxsize=task_queue_size)

        # Only in the main process do we maintain a futures dictionary and background thread.
        if current_process().name == 'MainProcess':
            self._futures: Dict[int, Future[TResult]] = {}
            self._task_counter = 0
            self._result_listener_thread = threading.Thread(target=self._result_listener, daemon=True)
            self._result_listener_thread.start()

    def _run_loop(self):
        """
        The worker process loop. It expects to receive (task_id, task) tuples.
        """
        while not self.stop_requested.is_set():
            item = self.tasks.get()
            if item is None:
                logger.debug(f"Worker {self.worker_id}: Stop signal received.")
                self.stop_requested.set()
                continue

            task_id, task = item
            try:
                result = self.handle_task(task)
                self.results.put((task_id, result))
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error handling task: {e}")
                self.results.put((task_id, e))
        self.cleanup()

    def handle_task(self, task: TTask) -> TResult:
        """
        Process a single task. Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement handle_task.")

    def _result_listener(self):
        """
        A background thread (running in the main process) that listens for results from the worker process
        and resolves the corresponding Future.
        """
        while True:
            task_id, result = self.results.get()
            future = self._futures.pop(task_id, None)
            if future:
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)

    def submit_task(self, task: TTask) -> Future[TResult]:
        """
        Submits a task for processing and returns a Future that can later be used to obtain the result.
        """
        if not hasattr(self, "_futures"):
            raise RuntimeError("submit_task should only be called from the main process.")
        future = Future[TResult]()
        task_id = self._task_counter
        self._task_counter += 1
        self._futures[task_id] = future
        self.tasks.put((task_id, task))
        return future

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

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-pickleable attributes
        state.pop('_result_listener_thread', None)
        state.pop('_futures', None)
        return state


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
