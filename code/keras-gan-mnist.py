# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop

from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Reshape
from keras.optimizers import SGD
from PIL import Image
import argparse
import math
import matplotlib.mlab as MLA
from keras.layers.normalization import BatchNormalization



def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=256))
    model.add(Activation('relu'))
    model.add(Dense(28*28*1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.add(Reshape((28,28,1), input_shape=(28*28,)))

    plot_model(model, show_shapes=True, to_file='gan/keras-gan-mnist-generator_model.png')
    return model


def discriminator_model():
    model = Sequential()
    model.add(Reshape((28*28, ), input_shape=(28,28,1)))
    model.add(Dense(units=256))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    plot_model(model, show_shapes=True, to_file='gan/keras-gan-mnist-discriminator_model.png')
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    plot_model(model, show_shapes=True, to_file='gan/keras-gan-mnist-gan_model.png')
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]

    return image

def train(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]

    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 1000 == 0:
                print "combine_images"
                image = combine_images(generated_images)
                image = image*127.5+127.5
                print image
                #调试阶段不生成图片
                Image.fromarray(image.astype(np.uint8)).save("gan/"+str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            #print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            #print("batch %d g_loss : %f" % (index, g_loss))


if __name__ == '__main__':
    train(128)




