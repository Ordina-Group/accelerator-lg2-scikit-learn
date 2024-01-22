import os
from typing import Optional

import skimage
import numpy as np
from matplotlib import pyplot as plt

IMAGE_ARRAY_DIMENSIONS = (128, 128, 3)


def read_image_classification_dataset(data_path: str) -> tuple[np.array, np.array]:
    """Read image dataset from disk.

    The returned arrays contain the images and labels in matching order. The image arrays
    are flattened for ease of use with scikit-learn models. The labels are integers, where
    0=car and 1=truck.

    """
    images = []
    labels = []

    class_directories = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for class_id, directory_name in enumerate(class_directories):
        directory_path = os.path.join(data_path.rstrip(), directory_name)
        file_names = os.listdir(directory_path)

        images.extend([skimage.io.imread(os.path.join(directory_path, file_name)).flatten() for file_name in file_names])
        labels.extend([class_id] * len(file_names))

    return np.array(images), np.array(labels)


def show_image_with_text(image: np.array, text: Optional[str] = None) -> None:
    """Show image with optional text above it.

    Will stop execution until window is closed.

    """
    plt.imshow(image.reshape(IMAGE_ARRAY_DIMENSIONS))
    plt.text(0, -5, text, fontsize=28)
    plt.show()
