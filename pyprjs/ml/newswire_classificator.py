# Import dataset Reuters
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

from tensorflow.keras.utils import to_categorical

from imdb_model import vectorize_sequences, data_grapher

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model = keras.Sequential([
    layers.Dense(64, activation = "relu"),
    layers.Dense(64, activation = "relu"),
    layers.Dense(46, activation = "softmax")
])

model.compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

# Validation data separation

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs = 9, batch_size = 512, validation_data= (x_val, y_val))

data_grapher(history)

print("PREDICTIONS:\n")
import numpy as np

predictions = model.predict(x_test)

print(predictions[0].shape)

print(np.argmax(predictions[0]))

