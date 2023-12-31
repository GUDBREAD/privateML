from tensorflow.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

#Data decoder example
#-------------------------------------
word_index = imdb.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = " ".join(reverse_word_index.get(i-3, "_") for i in train_data[0])
#print(decoded_review)
#-------------------------------------


# Data preprocessing -> multi-hot encoding
#----------------------------------------------------
import numpy as np

def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i,j] = 1.
    return(results)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


#print(x_train)  

y_train = np.asarray(train_labels).astype("float32")
y_test =np.asarray(test_labels).astype("float32")    

#----------------------------------------------------

# Importing keras, compiling model (Overfitting)
#----------------------------------------------------
from tensorflow import keras
from tensorflow.keras import layers

# model = keras.Sequential([
#     layers.Dense(16, activation = "relu"),
#     layers.Dense(16, activation = "relu"),
#     layers.Dense(1, activation = "sigmoid"),
# ])

# model.compile(optimizer = "rmsprop",
#               loss = "binary_crossentropy",
#               metrics = ["accuracy"])

# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]

#history = model.fit(partial_x_train,partial_y_train, epochs = 20, batch_size = 512, validation_data = (x_val,y_val))

#----------------------------------------------------
def data_grapher(history):
#Graphing data: LOSS

#----------------------------------------------------
    import matplotlib.pyplot as plt
    history_dict = history.history

    loss_values = history_dict["loss"]

    val_loss_values = history_dict["val_loss"]

    epochs = range(1, len(loss_values)+1)

    plt.plot(epochs, loss_values, "bo", label = "Training loss")
    plt.plot(epochs, val_loss_values, "b", label = "Validation loss")

    plt.title("Training + validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

    #----------------------------------------------------

    #Graphing data: ACCURACY

    #----------------------------------------------------

    plt.clf()

    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]



    plt.plot(epochs, acc, "bo", label = "Training acc")
    plt.plot(epochs, val_acc, "b", label = "Validation acc")

    plt.title("Training + Validation Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.show()

#Model 2 overfit correction
#----------------------------------------------------

# model2 = keras.Sequential([
#     layers.Dense(16, activation = "relu"),
#     layers.Dense(16, activation = "relu"),
#     layers.Dense(1, activation = "sigmoid"),
# ])

# model2.compile(optimizer = "rmsprop",
#               loss = "binary_crossentropy",
#               metrics = ["accuracy"])

# model2.fit(x_train, y_train, epochs = 4, batch_size = 512)

# results = model2.evaluate(x_test, y_test)

# print(results)
