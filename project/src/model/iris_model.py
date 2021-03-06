import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.python.keras.layers.advanced_activations import ReLU, Softmax
from tensorflow.python.keras.layers.core import Dense


class Iris_Model(tf.keras.Model):
    def __init__(self, n_inputs, n_output):
        super(Iris_Model, self).__init__()

        self.input_layer = Dense(n_inputs, activation=ReLU)
        self.dense_layer = Dense(128, activation=ReLU)
        self.output_layer = Dense(n_output, activation=Softmax)
    
    def call(self, inputs):

        output = self.input_layer(inputs)

        output = self.dense_layer(output)

        output = self.output_layer(output)

        return output