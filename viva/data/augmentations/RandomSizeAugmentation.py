import random
from typing import Tuple

import numpy as np

from viva.data.FaceLandmarkSeries import FaceLandmarkSeries
from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation


class RandomSizeAugmentation(BaseLandmarkAugmentation):
    def __init__(self,
                 min_pixel_change: float = 1.0,
                 max_pixel_change: float = 3.0,
                 image_size: int = 256,
                 clip: bool = True,
                 probability: float = 0.5):
        """
        Initialize the augmentation that simulates small size changes of the input image by scaling
        landmark coordinates. This is done by applying a per-frame scaling transformation about the image
        center (assumed to be at 0.5 in normalized coordinates).

        For an image of size `image_size`, a change of Δ pixels at the border (distance = image_size/2)
        corresponds to a scaling factor of:
            s = 1 + (Δ / (image_size/2)).

        For example, for image_size=256 and Δ=3, s = 1 ± (3/128).

        :param min_pixel_change: Minimum magnitude (in pixels) for the size change at the image border.
        :param max_pixel_change: Maximum magnitude (in pixels) for the size change at the image border.
        :param image_size: Size of the (square) input images.
        :param clip: If True, clip the augmented (x,y) coordinates to remain within [0, 1].
        :param probability: The probability (between 0 and 1) that the augmentation is applied.
        """
        self.min_pixel_change = min_pixel_change
        self.max_pixel_change = max_pixel_change
        self.image_size = image_size
        self.clip = clip
        self.probability = probability

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 series: FaceLandmarkSeries,
                 start_index: int,
                 end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random scaling to simulate size changes in the input image.
        Each frame in the sequence is scaled independently (i.e. the effective size jumps over time).

        The (x, y) coordinates of each landmark are transformed by:
            new_xy = (old_xy - center) * s + center,
        where s is the scaling factor computed from a random pixel shift (converted to normalized units)
        and center is assumed to be 0.5.

        :param x: Landmark data (numpy array) of shape (T, N, 3) with normalized coordinates in [0, 1].
        :param y: Labels corresponding to the landmarks.
        :param series: FaceLandmarkSeries containing all relevant series information.
        :param start_index: Start index of the current sequence inside the series.
        :param end_index: End index of the current sequence inside the series.
        :return: Tuple of augmented landmark data (x) and labels (y).
        """
        # Only apply the augmentation with the given probability.
        if random.random() > self.probability:
            return x, y

        # Work on a copy so as not to modify the original data.
        x_aug = x.copy()
        num_frames = x.shape[0]

        # For each frame, sample a scaling factor based on a random pixel change.
        # Note: The maximum effect of scaling is at the image border, which is at a normalized distance of 0.5.
        # So, to achieve a Δ pixels change at the border:
        #     (image_size/2) * (s - 1) = Δ  ==>  s = 1 + (Δ / (image_size/2))
        scales = np.empty(num_frames, dtype=np.float32)
        for t in range(num_frames):
            # Randomly choose a magnitude in pixels between min_pixel_change and max_pixel_change.
            delta_pixels = random.uniform(self.min_pixel_change, self.max_pixel_change)
            # Randomly decide whether to increase or decrease the size.
            if random.random() < 0.5:
                delta_pixels = -delta_pixels

            # Calculate the scaling factor.
            scale = 1 + (delta_pixels / (self.image_size / 2))
            scales[t] = scale

        # Reshape scales for broadcasting: (num_frames, 1, 1)
        scales = scales.reshape(-1, 1, 1)

        # Define the center of scaling (assumed to be the center of the image in normalized coordinates).
        center = 0.5

        # Apply the scaling to the x and y coordinates (first two channels); leave z unchanged.
        x_aug[..., :2] = (x_aug[..., :2] - center) * scales + center

        if self.clip:
            # Ensure that the (x, y) coordinates remain in [0, 1] after transformation.
            x_aug[..., :2] = np.clip(x_aug[..., :2], 0, 1)

        return x_aug, y
