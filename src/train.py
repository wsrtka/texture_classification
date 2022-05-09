"""Script used to train models for texture classification."""

import argparse

import seaborn as sns
import numpy as np

from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD

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

LR = 1e-2
MOMENTUM = 0.9
BATCH_SIZE = 128
NUM_EPOCHS = 60
INPUT_DIMS = (0, 0, 0)
NUM_CLASSES = 0

if args["model"] == "vgg":
    model = vgg.VGG16(INPUT_DIMS, NUM_CLASSES)

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
