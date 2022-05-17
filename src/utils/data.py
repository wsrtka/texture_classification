"""Module containing functions for data loading."""

import os
import pathlib

import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

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


def resize_images(img_dir: pathlib.Path, size: tuple[int], img_format: str = "JPEG"):
    """Recurrently resize all images in folder to given size.

    Args:
        img_dir (pathlib.Path): Path to image directory
        size (tuple[int]): Size of output images
        img_format (str, optional): Image format. Defaults to 'JPEG'.
    """
    for item in tqdm(os.listdir(img_dir), desc=str(img_dir)):
        # get full path of image
        item_path = img_dir / item
        # resize if image, go deeper if directory
        if os.path.isfile(item_path):
            try:
                im = Image.open(item_path)
                imResize = im.resize(size, Image.ANTIALIAS)
                imResize.save(item_path, img_format)
            except UnidentifiedImageError:
                continue
        elif os.path.isdir(item_path):
            resize_images(item_path, size)


if __name__ == "__main__":
    imdir, _ = get_dataset("dtd")
    resize_images(imdir, (300, 300))
