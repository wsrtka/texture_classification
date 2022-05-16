"""Script used to train models for texture classification."""

import argparse
import logging

import seaborn as sns
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.python.compiler.mlcompute import mlcompute

from src.models import vgg


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="vgg",
    choices=["vgg"],
    help="Type of model architecture to train.",
)
args = vars(parser.parse_args())

logger = logging.getLogger()
logger.info("Running on tensorflow v%s", tf.__version__)

tf.compat.v1.disable_eager_execution()
mlcompute.set_mlc_device(device_name="gpu")

logger.info("Apple MLC enabled: %s", mlcompute.is_apple_mlc_enabled())
logger.info("Compiled with Apple MLC: %s", mlcompute.is_tf_compiled_with_apple_mlc())
logger.info("Eager excecution: %s", tf.executing_eagerly())
logger.info("Listing logical devices:")
logger.info(tf.config.list_logical_devices())

# todo: move these options into parser arguments
LR = 1e-2
MOMENTUM = 0.9
BATCH_SIZE = 128
NUM_EPOCHS = 60
# todo: move this into __init__.py and adjust according to dataset
# (add option in argparser for this also)
INPUT_DIMS = (0, 0, 0)
NUM_CLASSES = 0

if args["model"] == "vgg":
    model = vgg.VGG16(INPUT_DIMS, NUM_CLASSES)

model.summary()

optimizer = SGD(lr=LR, momentum=MOMENTUM, decay=LR / NUM_EPOCHS)
model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)

training = model.fit_generator()
preds = model.predict()
print(classification_report())

xs = np.arange(0, NUM_CLASSES)

sns.lineplot(x=xs, y=training.history["loss"], title="loss")
sns.lineplot(x=xs, y=training.history["val_loss"], title="val_loss")
sns.lineplot(x=xs, y=training.history["accuracy"], title="accuracy")
sns.lineplot(x=xs, y=training.history["val_accuracy"], title="val_accuracy")
