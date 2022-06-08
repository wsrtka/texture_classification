"""Module containing CNNs for texture classification."""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    Rescaling,
    AveragePooling2D,
    Input,
    Lambda,
    BatchNormalization,
    Reshape,
    Activation,
    concatenate,
    add,
)
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

from src.utils.wavelet import Wavelet, Wavelet_out_shape


class AlexNet(tf.keras.Model):
    """Implementation of AlexNet architecture."""

    def __init__(self, input_dims, num_classes):
        super().__init__()

        self.rescale = Rescaling(1.0 / 255)

        # conv layer 1
        self.conv1 = Sequential(
            [
                Conv2D(
                    filters=96,
                    kernel_size=11,
                    strides=4,
                    activation="relu",
                    padding="same",
                    input_shape=input_dims,
                ),
                MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            ]
        )

        # conv layer 2
        self.conv2 = Sequential(
            [
                Conv2D(filters=256, kernel_size=5, activation="relu", padding="same"),
                MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            ]
        )

        # conv layer 3
        self.conv3 = Sequential(
            [
                Conv2D(filters=384, kernel_size=3, activation="relu", padding="same"),
                Conv2D(filters=384, kernel_size=3, activation="relu", padding="same"),
                Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"),
                MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
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
        x = self.classifier(x)

        return x


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


def get_wavelet_cnn_model(input_dims, num_classes):

    input_ = Input(input_dims, name="the_input")
    # wavelet = Lambda(Wavelet, name='wavelet')
    wavelet = Lambda(Wavelet, Wavelet_out_shape, name="wavelet")
    input_l1, input_l2, input_l3 = wavelet(input_)

    conv_1 = Conv2D(64, kernel_size=(3, 3), padding="same", name="conv_1")(input_l1)
    norm_1 = BatchNormalization(name="norm_1")(conv_1)
    relu_1 = Activation("relu", name="relu_1")(norm_1)

    conv_1_2 = Conv2D(
        64, kernel_size=(3, 3), strides=(2, 2), padding="same", name="conv_1_2"
    )(relu_1)
    norm_1_2 = BatchNormalization(name="norm_1_2")(conv_1_2)
    relu_1_2 = Activation("relu", name="relu_1_2")(norm_1_2)

    # level two decomposition starts
    conv_a = Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="conv_a")(
        input_l2
    )
    norm_a = BatchNormalization(name="norm_a")(conv_a)
    relu_a = Activation("relu", name="relu_a")(norm_a)

    # concate level one and level two decomposition
    concate_level_2 = concatenate([relu_1_2, relu_a])
    conv_2 = Conv2D(128, kernel_size=(3, 3), padding="same", name="conv_2")(
        concate_level_2
    )
    norm_2 = BatchNormalization(name="norm_2")(conv_2)
    relu_2 = Activation("relu", name="relu_2")(norm_2)

    conv_2_2 = Conv2D(
        128, kernel_size=(3, 3), strides=(2, 2), padding="same", name="conv_2_2"
    )(relu_2)
    norm_2_2 = BatchNormalization(name="norm_2_2")(conv_2_2)
    relu_2_2 = Activation("relu", name="relu_2_2")(norm_2_2)

    # level three decomposition starts
    conv_b = Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="conv_b")(
        input_l3
    )
    norm_b = BatchNormalization(name="norm_b")(conv_b)
    relu_b = Activation("relu", name="relu_b")(norm_b)

    conv_b_2 = Conv2D(128, kernel_size=(3, 3), padding="same", name="conv_b_2")(relu_b)
    norm_b_2 = BatchNormalization(name="norm_b_2")(conv_b_2)
    relu_b_2 = Activation("relu", name="relu_b_2")(norm_b_2)

    # concate level two and level three decomposition
    concate_level_3 = concatenate([relu_2_2, relu_b_2])
    conv_3 = Conv2D(256, kernel_size=(3, 3), padding="same", name="conv_3")(
        concate_level_3
    )
    norm_3 = BatchNormalization(name="nomr_3")(conv_3)
    relu_3 = Activation("relu", name="relu_3")(norm_3)

    conv_3_2 = Conv2D(
        256, kernel_size=(3, 3), strides=(2, 2), padding="same", name="conv_3_2"
    )(relu_3)
    norm_3_2 = BatchNormalization(name="norm_3_2")(conv_3_2)
    relu_3_2 = Activation("relu", name="relu_3_2")(norm_3_2)

    conv_4 = Conv2D(256, kernel_size=(3, 3), padding="same", name="conv_4")(relu_3_2)
    norm_4 = BatchNormalization(name="norm_4")(conv_4)
    relu_4 = Activation("relu", name="relu_4")(norm_4)

    conv_4_2 = Conv2D(
        256, kernel_size=(3, 3), strides=(2, 2), padding="same", name="conv_4_2"
    )(relu_4)
    norm_4_2 = BatchNormalization(name="norm_4_2")(conv_4_2)
    relu_4_2 = Activation("relu", name="relu_4_2")(norm_4_2)

    conv_5_1 = Conv2D(128, kernel_size=(3, 3), padding="same", name="conv_5_1")(
        relu_4_2
    )
    norm_5_1 = BatchNormalization(name="norm_5_1")(conv_5_1)
    relu_5_1 = Activation("relu", name="relu_5_1")(norm_5_1)

    pool_5_1 = AveragePooling2D(
        pool_size=(7, 7), strides=1, padding="same", name="avg_pool_5_1"
    )(relu_5_1)
    flat_5_1 = Flatten(name="flat_5_1")(pool_5_1)

    fc_5 = Dense(2048, name="fc_5")(flat_5_1)
    norm_5 = BatchNormalization(name="norm_5")(fc_5)
    relu_5 = Activation("relu", name="relu_5")(norm_5)
    drop_5 = Dropout(0.5, name="drop_5")(relu_5)

    fc_6 = Dense(2048, name="fc_6")(drop_5)
    norm_6 = BatchNormalization(name="norm_6")(fc_6)
    relu_6 = Activation("relu", name="relu_6")(norm_6)
    drop_6 = Dropout(0.5, name="drop_6")(relu_6)

    output = Dense(num_classes, activation="softmax", name="fc_7")(drop_6)

    model = Model(inputs=input_, outputs=output)

    return model
