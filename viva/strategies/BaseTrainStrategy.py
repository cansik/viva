from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, List, Dict, TypeVar, Generic, Callable, Type, Union

import pytorch_lightning as pl

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset


@dataclass
class BaseTrainOptions:
    # training
    batch_size: int = 32
    max_epochs: int = 100
    num_workers: int = 4
    mixed: bool = False

    early_stopping: bool = True
    early_stopping_patience: int = 5

    # dataset
    block_size: int = 15

    # output
    log_dir: str = "runs"

    # debug
    profile: bool = False

    def overwrite_options(self, training_options: List[str]) -> None:
        # Convert name=value pairs to a dictionary
        options_dict: Dict[str, Any] = {}
        for option in training_options:
            if "=" not in option:
                raise ValueError(f"Invalid format for option '{option}', expected 'name=value'.")
            name, value = option.split("=", 1)
            options_dict[name] = value

        # Convert string values to their correct types based on the dataclass fields
        for field in fields(self):
            if field.name in options_dict:
                field_type = field.type
                try:
                    setattr(self, field.name, field_type(options_dict[field.name]))
                except ValueError as e:
                    raise ValueError(f"Invalid type for field '{field.name}': {e}")


T = TypeVar("T", bound=BaseTrainOptions)


class BaseTrainStrategy(ABC, Generic[T]):
    @property
    @abstractmethod
    def options(self) -> T:
        pass

    @abstractmethod
    def create_lighting_module(self) -> pl.LightningModule:
        pass

    @property
    @abstractmethod
    def train_dataset_type(self) -> Union[Type[FaceLandmarkDataset], Callable[..., FaceLandmarkDataset]]:
        pass

    @property
    @abstractmethod
    def test_dataset_type(self) -> Union[Type[FaceLandmarkDataset], Callable[..., FaceLandmarkDataset]]:
        pass
