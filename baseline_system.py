"""
baseline code system
database:reuters
Author: Aaron Lee
Date: 28/03/2019
Student Id: 300422249
"""

import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout,Embedding,Conv1D,GlobalMaxPooling1D
import matplotlib.pyplot as plt

def main():
    # model parameters:
    maxlen = 400
    max_words = 10000
    batch_size = 32
    epochs = 20
    embedding_dims = 50
    cnn_filters = 100
    cnn_kernel_size = 5
    dense_hidden_dims = 200

    # 1. Loading started
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.2)
    word_index = reuters.get_word_index(path="reuters_word_index.json")

    num_classes = max(y_train) + 1
    # 2. pad_sequences
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # 3. Build CNN model...
    model = Sequential()
    model.add(Embedding(max_words, embedding_dims, input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(cnn_filters, cnn_kernel_size, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(dense_hidden_dims, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.summary()

    # 4. compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    # 5.  train model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

    # 6. evaluate model
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size, verbose=1)
    print('Test loss:{}\nTest accuracy:{}'.format(loss_and_metrics[0], loss_and_metrics[1]))

    # Create a graph of accuracy and loss over time
    history_dict = history.history
    history_dict.keys()

    acc = history_dict['categorical_accuracy']
    val_acc = history_dict['val_categorical_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
  main()
