# from project.src.model.iris_model import Iris_Model
import tensorflow as tf

from tensorflow import keras

import os

from project.src.data.preprocess_data import x_train
from project.src.data.preprocess_data import y_train
from project.src.data.preprocess_data import x_validate
from project.src.data.preprocess_data import y_validate

from project.src.data.preprocess_data import x_dim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# model = Iris_Model(4,3)

train_values = tf.convert_to_tensor(x_train)
train_labels = tf.keras.utils.to_categorical(y_train, 3)

validate_values = tf.convert_to_tensor(x_validate)
validate_labels = tf.keras.utils.to_categorical(y_validate, 3)

# print(train_labels)
model = keras.Sequential([
    keras.Input(shape=4),
    # hidden layer
    keras.layers.Dense(8, activation='relu'),
    # output
    keras.layers.Dense(3)
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

print("shape of train values : {}",train_values.shape)
print("shape of train labels : {}", train_labels.shape)
model.fit(x_train, train_labels, epochs=5, verbose=2)

model.evaluate(x_validate, validate_labels, verbose=2)