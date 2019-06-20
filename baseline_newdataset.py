"""
baseline code system  - new Datast(PanNewData.csv)
database:Twitter feed data--label: bot, male, female.
Author: Aaron Lee
Date: 28/04/2019
Student Id: 300422249
"""
import pandas
import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing import text, sequence
from sklearn import model_selection, preprocessing

# running_newdataset
def running_newdataset(modelname):
    #1.load data  panData
    data = pandas.read_csv("PanNewData.csv", encoding="latin-1")
    # split the dataset into training and validation datasets
    train_x, test_x, train_y, test_y = model_selection.train_test_split(data['text'], data['label'],test_size=0.33, shuffle=False)

    #2. label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)

    #3. create a tokenizer
    token = text.Tokenizer()
    token.fit_on_texts(data['text'])
    word_index = token.word_index
    maxlen = 40
    vocab_size = len(word_index) + 1

    keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,split=' ', char_level=False, oov_token=None, document_count=0)

    #4. convert text to sequence of tokens an
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=maxlen)
    test_seq_x = sequence.pad_sequences(token.texts_to_sequences(test_x), maxlen=maxlen)
    bulidModel(modelname,train_seq_x,test_seq_x,train_y,test_y,vocab_size)

def bulidModel(modelname,train_seq_x,test_seq_x,train_y,test_y,vocab_size):
    maxlen = 40
    max_words = 10000
    batch_size = 2000
    epochs = 20
    embedding_dims = 50

    if (modelname == 'cnn'):
        # 6. define model--CNN
        print("Creating CNN model")
        model = Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dims, input_length=maxlen))
        model.add(layers.Conv1D(128, 5, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()

    elif(modelname=='rnn'):
        print("Creating LSTM model")
        e_init = keras.initializers.RandomUniform(-0.01, 0.01, seed=1)
        init = keras.initializers.glorot_uniform(seed=1)
        simple_adam = keras.optimizers.Adam()
        model = keras.models.Sequential()
        model.add(keras.layers.embeddings.Embedding(input_dim=vocab_size, output_dim=embedding_dims,embeddings_initializer=e_init, mask_zero=True))
        model.add(keras.layers.LSTM(units=100, kernel_initializer=init, dropout=0.2, recurrent_dropout=0.2))  # 100 memory
        model.add(keras.layers.Dense(units=1, kernel_initializer=init, activation='sigmoid'))
        model.summary()

     # 7. compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(train_seq_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.1)
    # 8. evaluate model
    loss_and_metrics = model.evaluate(test_seq_x, test_y, verbose=0)
    print('Test loss:{}\nTest accuracy:{}'.format(loss_and_metrics[0], loss_and_metrics[1]))
    showHistory(history)

def showHistory(history):
    # 9. Create a graph of accuracy and loss over time
    history_dict = history.history
    history_dict.keys()

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
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
    running_newdataset('rnn')
    # cnn  rnn  running_newdataset('rnn')