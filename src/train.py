"""Script used to train models for texture classification."""

import argparse

import seaborn as sns
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

# pylint: disable=no-name-in-module
from src.models import VGG16
from src.utils.data import get_dataset

# from src.utils.visualize import show_dataset


# initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="vgg",
    choices=["vgg"],
    help="Type of model architecture to train.",
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="dtd",
    choices=["dtd"],
    help="Dataset to train.",
)
args = vars(parser.parse_args())

# initialize logger
logger = tf.get_logger()
logger.setLevel("INFO")
logger.info("Running on tensorflow v%s", tf.__version__)

# tensorflow settings
# tf.compat.v1.disable_eager_execution()

# log tensorflow settings
logger.info("Eager excecution: %s", tf.executing_eagerly())
logger.info("Listing logical devices:")
logger.info(tf.config.list_logical_devices())

# todo: move these options into parser arguments
LR = 1e-2
MOMENTUM = 0.9
BATCH_SIZE = 64
NUM_EPOCHS = 60
# todo: move this into __init__.py and adjust according to dataset
# (add option in argparser for this also)
INPUT_DIMS = (150, 150, 3)
NUM_CLASSES = 47

# choose and print model
if args["model"] == "vgg":
    model = VGG16(INPUT_DIMS, NUM_CLASSES)

# load dataset
data_dir, img_count = get_dataset(args["dataset"])
logger.info("Loaded dataset with %d images", img_count)

if args["dataset"] == "dtd":
    data_dir = data_dir / "images"

# split dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=INPUT_DIMS[:2],
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=INPUT_DIMS[:2],
    batch_size=BATCH_SIZE,
)

# show_dataset(train_ds)

# configure dataset for performance
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# prepare optimizer
optimizer = SGD(lr=LR, momentum=MOMENTUM, decay=LR / NUM_EPOCHS)

# compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)

# model.summary()

# create callback
cp_callback = ModelCheckpoint(
    filepath="checkpoints/cp-{epoch:04d}.ckpt",
    verbose=1,
    save_weights_only=True,
    save_freq=5 * BATCH_SIZE,
)

# train model
training = model.fit(
    train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=[cp_callback]
)

# save model
model.save(f'models/{args["model"]}')

# plot results
xs = np.arange(0, NUM_EPOCHS)

sns.lineplot(x=xs, y=training.history["loss"], title="loss")
sns.lineplot(x=xs, y=training.history["val_loss"], title="val_loss")
sns.lineplot(x=xs, y=training.history["accuracy"], title="accuracy")
sns.lineplot(x=xs, y=training.history["val_accuracy"], title="val_accuracy")
