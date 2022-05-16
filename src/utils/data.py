"""Module containing functions for data loading."""

# import os
import pathlib

# import PIL
# import PIL.Image
# import numpy as np
import tensorflow as tf

from src.utils import URLS


def get_dataset(name):
    """Get dataset from url.

    Args:
        name (str): name of dataset

    Returns:
        tuple: directory and size of downloaded dataset
    """
    dataset_url = URLS[name]
    data_dir = tf.keras.utils.get_file(origin=dataset_url, fname=name, untar=True)
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob("**/*.jpg")))
    return data_dir, image_count


if __name__ == "__main__":
    print(get_dataset("dtd"))
