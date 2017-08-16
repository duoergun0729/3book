from sklearn.feature_extraction.text import CountVectorizer
import os
#from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
#import matplotlib.pyplot as plt
import numpy as np
#from sklearn import svm
#from sklearn.feature_extraction.text import TfidfTransformer
#import tensorflow as tf
#import tflearn
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.conv import conv_1d, global_max_pool
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.merge_ops import merge
#from tflearn.layers.estimator import regression
#from tflearn.data_utils import to_categorical, pad_sequences
#from sklearn.neural_network import MLPClassifier
#from tflearn.layers.normalization import local_response_normalization
#from tensorflow.contrib import learn


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import numpy as np
from sklearn import metrics
from keras.layers import Merge


max_features=500
max_document_length=100
batch_size = 32
embedding_dims = 100
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2
num_classes=2



def load_one_file(filename):
    x=""
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            line = line.strip('\r')
            x+=line
    return x

def load_files_from_dir(rootdir):
    x=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v=load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    ham=[]
    spam=[]
    for i in range(1,2):
        path="../data/mail/enron%d/ham/" % i
        print "Load %s" % path
        ham+=load_files_from_dir(path)
        path="../data/mail/enron%d/spam/" % i
        print "Load %s" % path
        spam+=load_files_from_dir(path)
    return ham,spam

def get_features_by_wordbag():
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print vectorizer
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    return x,y



def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print "NB and wordbag"
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

def do_svm_wordbag(x_train, x_test, y_train, y_test):
    print "SVM and wordbag"
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)




def do_cnn_wordbag_tflearn(trainX, testX, trainY, testY):
    global max_document_length
    print "CNN and tf"

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_document_length], name='input')
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100,run_id="spam")


def do_cnn_wordbag_keras(x_train, x_test, y_train, y_test):
    #global max_document_length
    #global max_features
    print "CNN and keras"

    x_train = sequence.pad_sequences(x_train, maxlen=max_document_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_document_length)

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()

    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=max_document_length))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    y_pred = np.argmax(model.predict(x_test),axis=1)
    y_true = np.argmax(y_test,axis=1)
    cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
    print cm

    print('metrics.accuracy_score:', metrics.accuracy_score(y_true,y_pred))
    print('metrics.precision_score:', metrics.precision_score(y_true,y_pred))
    print('metrics.recall_score:', metrics.recall_score(y_true,y_pred))


def do_cnn_wordbag_keras_345(x_train, x_test, y_train, y_test):
    #global max_document_length
    #global max_features
    print "CNN and keras"

    x_train = sequence.pad_sequences(x_train, maxlen=max_document_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_document_length)

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    embedding_layer=Embedding(max_features,
                        embedding_dims,
                        input_length=max_document_length)

    print('Build model...')


    model1 = Sequential()

    model1.add(embedding_layer)
    model1.add(Dropout(0.2))

    model1.add(Conv1D(filters,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model1.add(GlobalMaxPooling1D())

    model2 = Sequential()

    model2.add(embedding_layer)
    model2.add(Dropout(0.2))

    model2.add(Conv1D(filters,
                     4,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model2.add(GlobalMaxPooling1D())


    model3 = Sequential()

    model3.add(embedding_layer)
    model3.add(Dropout(0.2))

    model3.add(Conv1D(filters,
                     5,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model3.add(GlobalMaxPooling1D())

    merged = Merge([model1, model2, model3], mode='concat')

    model = Sequential()
    model.add(merged)
    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(2))
    model.add(Activation('sigmoid'))


    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    y_pred = np.argmax(model.predict(x_test),axis=1)
    y_true = np.argmax(y_test,axis=1)
    cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
    print cm

    print('metrics.accuracy_score:', metrics.accuracy_score(y_true,y_pred))
    print('metrics.precision_score:', metrics.precision_score(y_true,y_pred))
    print('metrics.recall_score:', metrics.recall_score(y_true,y_pred))


if __name__ == "__main__":
    print "Hello spam-mail"
    print "get_features_by_wordbag"
    x,y=get_features_by_wordbag()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)

    #CNN
    do_cnn_wordbag_keras_345(x_train, x_test, y_train, y_test)

