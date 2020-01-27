import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import json
import os.path
import parse_logs


# GPU configuration
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def parse_file_json(filename, array_data, array_labels):
    with open(filename) as json_file:
        data = json.load(json_file)
        for p in data:
            array_data.append(p["game"])
            array_labels.append(p["status"])


# SET DATA
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "data\huge_games.json")
train_data = []
train_labels = []

train_data = parse_logs.parse_file('fool.log', train_data)
#parse_file_json(path, train_data, train_labels) # Load training data
train_data = np.array(train_data)
train_labels = np.array(train_labels)


# Vectorize input data is only necessary when the input arrays' lenght is not the same.
# Ex: dataset of twits: different length (number of words per twit). Dataset of tictatoe games: same length (board has
# always same size).
def vectorize_sequences(sequences, dimension=24389):
    """
    One-hot encode your lists to turn them into vectors of 0s and 1s.
    :param sequences: data input sequences
    :param dimension: number of possible different values
    :return: vectorized matrix
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)

x_train = train_data
#x_test = test_data


# one_hot_train_labels = to_categorical(train_labels)  # Uncomment if loss='categorical_crossentropy' at compiling model
# one_hot_test_labels = to_categorical(test_labels)  # Uncomment if loss='categorical_crossentropy' at compiling model
one_hot_train_labels = train_labels  # Uncomment if loss='sparse_categorical_crossentropy' at compiling model
# one_hot_test_labels = test_labels  # Uncomment if loss='sparse_categorical_crossentropy' at compiling model


# Separate x units of data to monitor the training
number_data = 900
x_val = x_train[:number_data]
partial_x_train = x_train[number_data:]
y_val = one_hot_train_labels[:number_data]
partial_y_train = one_hot_train_labels[number_data:]


# Building and training the neural network
model = models.Sequential()
model.add(layers.Dense(400, activation='relu', input_shape=(9,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(125, activation='relu'))
model.add(layers.Dense(125, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(partial_x_train,partial_y_train,epochs=40,batch_size=512,validation_data=(x_val, y_val))

# Save model to file

#model.save('modeljson_norand.h5')  # Uncomment when layer configuration is defined

# Monitoring loss and accuracy while training
# Uncomment while testing the configuration

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
