"""This file contains the sructure, instanciation
and compilation of the vgecocup model. If you want to
modify the CNN, you have to modify the function
'get_vgecocup'
"""
import tensorflow as tf

from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16


def get_vgecocup():
    """Get the VGEcocup model used for the ecocup detection
    project

    Returns:
        tf.keras.Model: VGEcocup Model
    """
    vggmodel = VGG16(weights='imagenet', include_top=True)

    for layers in (vggmodel.layers)[:15]:
        layers.trainable = False

    dense_layer2 = vggmodel.layers[-2].output
    prediction_layer = Dense(units=2, activation="softmax")(dense_layer2)

    vgecocup = Model(inputs=vggmodel.input, outputs=prediction_layer)
    optimizer = Adam(learning_rate=0.0001)
    vgecocup.compile(optimizer=optimizer,
                     loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

    return vgecocup
