"""This file is used to train the model returned by
get_vgecocup function with the data created by create_data
file
"""
import numpy as np

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from vgecocup import get_vgecocup


train_images_array = np.load("train_images_full.npy")
train_labels_array = np.load("train_labels_full.npy")

train_labels_array = to_categorical(train_labels_array, dtype="int")

X_train_model, X_test_model, y_train_model, y_test_model = train_test_split(
    train_images_array, train_labels_array, test_size=0.10)

vgecocup = get_vgecocup()

hist = vgecocup.fit(
    x=X_train_model,
    y=y_train_model,
    verbose=1,
    batch_size=10,
    steps_per_epoch=10,
    epochs=50,
)

vgecocup.save("models/vgecocup_version2")
