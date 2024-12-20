import cv2
import numpy as np


def resize_image_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """
    Resizes an image to fit within the specified width and height while maintaining its aspect ratio.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        max_width (int): The maximum allowed width for the resized image.
        max_height (int): The maximum allowed height for the resized image.

    Returns:
        np.ndarray: The resized image with its aspect ratio maintained.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The input must be a NumPy array representing the image.")

    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]

    # Calculate the scaling factors for width and height
    width_scale = max_width / original_width
    height_scale = max_height / original_height

    # Use the smaller scale to maintain the aspect ratio
    scale = min(width_scale, height_scale)

    # Calculate the new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image using OpenCV's resize function
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image


def annotate_landmarks(
        image: np.ndarray, landmarks: np.ndarray, color: tuple = (0, 255, 0), marker_size: int = 5
) -> np.ndarray:
    """
    Annotates normalized landmarks as markers on an image.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        landmarks (np.ndarray): A normalized (n, 3) array of landmarks, where each row is (x, y, visibility).
        color (tuple): The color of the markers in BGR format (default is green).
        marker_size (int): The size of the markers (default is 5).

    Returns:
        np.ndarray: The annotated image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The input must be a NumPy array representing the image.")

    if not isinstance(landmarks, np.ndarray) or landmarks.shape[1] < 3:
        raise ValueError("Landmarks must be a NumPy array of shape (n, >=3).")

    # Get image dimensions
    height, width = image.shape[:2]

    # Iterate through landmarks and annotate them on the image
    for x_norm, y_norm in landmarks[:, :2]:
        x = int(x_norm * width)
        y = int(y_norm * height)
        cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=2)

    return image
