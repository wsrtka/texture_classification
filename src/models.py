"""Module containing CNNs for texture classification."""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    Rescaling,
)
from tensorflow.keras.models import Sequential


class VGG16(tf.keras.Model):
    """Implementation of VGG16 architecture."""

    def __init__(self, input_dims, num_classes):
        super().__init__()

        self.rescale = Rescaling(1.0 / 255)

        # conv layer 1
        self.conv1 = Sequential(
            [
                Conv2D(
                    filters=64,
                    kernel_size=3,
                    activation="relu",
                    padding="same",
                    input_shape=input_dims,
                ),
                Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
                MaxPool2D(strides=(2, 2)),
            ]
        )

        # conv layer 2
        self.conv2 = Sequential(
            [
                Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"),
                Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"),
                MaxPool2D(strides=(2, 2)),
            ]
        )

        # conv layer 3
        self.conv3 = Sequential(
            [
                Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"),
                Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"),
                Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"),
                MaxPool2D(strides=(2, 2)),
            ]
        )

        # conv layer 4
        self.conv4 = Sequential(
            [
                Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
                Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
                Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
                MaxPool2D(strides=(2, 2)),
            ]
        )

        # conv layer 5
        self.conv5 = Sequential(
            [
                Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
                Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
                Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
                MaxPool2D(strides=(2, 2)),
            ]
        )

        # classifier
        self.classifier = Sequential(
            [
                Flatten(),
                Dense(4096, activation="relu"),
                Dropout(0.5),
                Dense(4096, activation="relu"),
                Dropout(0.5),
                Dense(num_classes, activation="softmax"),
            ]
        )

    def call(self, inputs):
        """Get prediction from model."""
        x = self.rescale(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)

        return x
