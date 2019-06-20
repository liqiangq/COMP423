"""
baseline code system
database:reuters
Author: Aaron Lee
Date: 28/03/2019
Student Id: 300422249
"""
import keras
from keras.datasets import reuters
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Embedding,Conv1D,GlobalMaxPooling1D,Activation
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer

# running_retuter
def running_retuter(modelname):
    maxlen = 400
    max_words = 10000

    # 1. Loading started
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.2)

    word_index = reuters.get_word_index(path="reuters_word_index.json")
    num_classes = np.max(y_train) + 1

    # 2. pad_sequences
    keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)

    if (modelname == 'cnn'):
       x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen)
       x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen)
       y_train = keras.utils.to_categorical(y_train, num_classes)
       y_test = keras.utils.to_categorical(y_test, num_classes)

    elif(modelname == 'nn'):
        tokenizer = Tokenizer(num_words=max_words)
        x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
        x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    bulidModel(modelname, num_classes, x_test, y_test, x_train, y_train)


def bulidModel(modelname,num_classes, x_test, y_test,x_train, y_train):
    # 3. Build CNN model...
    maxlen = 400
    max_words = 10000
    batch_size = 32
    epochs = 10
    embedding_dims = 50
    cnn_filters = 100
    cnn_kernel_size = 5
    dense_hidden_dims = 200

    if (modelname=='cnn'):
        print('Building CNN model...')
        model = Sequential()
        model.add(Embedding(max_words, embedding_dims, input_length=maxlen))
        model.add(Dropout(0.2))
        model.add(Conv1D(cnn_filters, cnn_kernel_size, padding='valid', activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(dense_hidden_dims, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='sigmoid'))
        model.summary()

    elif(modelname=='nn'):
        print('Building NN model...')
        model = Sequential()
        model.add(Dense(512, input_shape=(max_words,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
    history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=2,validation_split=0.1)
    evaluate(model, history, x_test, y_test)


def evaluate(model,history,x_test,y_test):
    # 6. evaluate model
    batch_size = 32
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

    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # cnn code
    running_retuter('cnn')
    # nn code running_retuter('nn')
