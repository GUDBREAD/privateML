from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

print(train_data.shape) 

# Normalize data (?) (substracting mean and dividing by standard deviation)

mean = train_data.mean(axis = 0)
print(train_data)
train_data -= mean 
print("------------------------------------------------")
print(train_data)
std = train_data.std(axis = 0)

train_data /= std
test_data -= mean
test_data /= std

def build_model ():
    model = keras.Sequential([
        layers.Dense(64, activation = "relu"),
        layers.Dense(64, activation = "relu"),
        layers.Dense(1) 
    ])
    model.compile(optimizer = "rmsprop", loss = "mse", metrics = "mae" )

    return model

# Neural network version 1: ------------------
# k = 4

# num_val_samples = len(train_data)//k

# num_epochs = 100

# all_scores = []

# for i in range(k):
#     print(f"Processing fold #{i}")

#     val_data = train_data[i* num_val_samples : (i+1) * num_val_samples]
#     val_targets = train_targets[i* num_val_samples : (i+1) * num_val_samples]

#     partial_train_data = np.concatenate(
#         [train_data[:i*num_val_samples], 
#         train_data[(i+1) * num_val_samples:]], 
#         axis = 0
#         )    
#     partial_train_targets = np.concatenate(
#         [train_targets[:i*num_val_samples], 
#         train_targets[(i+1) * num_val_samples:]], 
#         axis = 0
#         )
    
#     model = build_model()

#     model.fit(partial_train_data, partial_train_targets, epochs = num_epochs, batch_size = 16, verbose = 0)

#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)

#     all_scores.append(val_mae)


# print(all_scores)

# Neural network version 2: --------------------------------------

# k = 4

# num_val_samples = len(train_data)//k

# num_epochs = 500

# all_mae_histories = []

# for i in range(k):
#     print(f"Processing fold #{i}")

#     val_data = train_data[i* num_val_samples : (i+1) * num_val_samples]
#     val_targets = train_targets[i* num_val_samples : (i+1) * num_val_samples]

#     partial_train_data = np.concatenate(
#         [train_data[:i*num_val_samples], 
#         train_data[(i+1) * num_val_samples:]], 
#         axis = 0
#         )    
#     partial_train_targets = np.concatenate(
#         [train_targets[:i*num_val_samples], 
#         train_targets[(i+1) * num_val_samples:]], 
#         axis = 0
#         )
    
#     model = build_model()

#     history = model.fit(partial_train_data, partial_train_targets, validation_data = (val_data, val_targets), epochs = num_epochs, batch_size = 16)

#     #val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
#     mae_history = history.history["val_mae"]

#     all_mae_histories.append(mae_history)


# print(all_mae_histories)

# # Compile graph

# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# import matplotlib.pyplot as plt
# truncated_mae = average_mae_history[10:]
# plt.plot(range(1, len(truncated_mae)+1), truncated_mae)

# plt.xlabel("Epochs")
# plt.ylabel("Validation MAE")
# plt.show()



model = build_model()

model.fit(train_data, train_targets, epochs = 130, batch_size = 16)

test_mse, test_mae = model.evaluate(test_data, test_targets)

print(test_mae)