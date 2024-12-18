from abc import ABC, abstractmethod

from rich.console import Console


class VivaBaseMode(ABC):
    def __init__(self, console: Console):
        self.console = console

    @abstractmethod
    def run(self):
        pass
