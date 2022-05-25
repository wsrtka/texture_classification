"""Module used for visualization of outcomes."""

import matplotlib.pyplot as plt


def show_dataset(dataset):
    """Visualize 10 example images from dataset.

    Args:
        dataset (tensorflow.data.Dataset): Dataset containing images.
    """
    plt.figure(figsize=(10, 10))
    class_names = dataset.class_names
    for images, labels in dataset.take(1):
        for i in range(9):
            _ = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
