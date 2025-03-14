import logging
import threading
from typing import Optional, Generic

from viva.worker.types import TResult

logger = logging.getLogger(__name__)


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
