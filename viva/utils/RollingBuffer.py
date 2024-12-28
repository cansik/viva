from typing import Tuple

import numpy as np


class RollingBuffer:
    def __init__(self, block_size: int, feature_shape: Tuple[int, ...]):
        """
        Initialize a rolling buffer with a fixed block size.

        :param block_size: Number of samples in the buffer.
        :param feature_shape: Shape of each feature sample (e.g., landmark dimensions).
        """
        self.block_size = block_size
        self.feature_shape = feature_shape
        self.buffer = np.zeros((block_size, *feature_shape), dtype=np.float32)
        self.index = 0

    def add(self, sample: np.ndarray):
        """
        Add a new sample to the rolling buffer, replacing the oldest sample if the buffer is full.

        :param sample: New sample to add, must match the feature shape.
        """
        if sample.shape != self.feature_shape:
            raise ValueError(f"Sample shape {sample.shape} does not match buffer feature shape {self.feature_shape}.")

        self.buffer[self.index] = sample
        self.index = (self.index + 1) % self.block_size

    def get(self) -> np.ndarray:
        """
        Get the current state of the rolling buffer.

        :return: A view of the buffer with the most recent samples in order.
        """
        return np.roll(self.buffer, -self.index, axis=0)
