# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import scipy.io
from keras.utils.generic_utils import Progbar
from keras.utils import plot_model


def generator_model_old():
    model = Sequential()
    model.add(Dense(input_dim=200, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*8*8))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((8, 8, 128), input_shape=(128*8*8,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    #plot_model(model, show_shapes=True, to_file='keras-dcgan-svhn/keras-dcgan-generator_model.png')
    return model

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))

    model.add(Activation('tanh'))
    model.add(Dense(1024*4*4))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((4, 4, 1024), input_shape=(1024*4*4,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(512, (2, 2),padding='same'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(256, (5, 5),padding='same'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (5, 5),padding='same'))

    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    #plot_model(model, show_shapes=True, to_file='keras-dcgan-svhn/keras-dcgan-generator_model.png')

    model.summary()

    return model


def discriminator_model_old():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(32, 32, 3))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #plot_model(model, show_shapes=True, to_file='keras-dcgan-svhn/keras-dcgan-discriminator_model.png')
    return model

def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(32, 32, 3))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #plot_model(model, show_shapes=True, to_file='keras-dcgan-svhn/keras-dcgan-discriminator_model.png')
    return model

def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)

    #plot_model(model, show_shapes=True, to_file='keras-dcgan-svhn/keras-dcgan-gan_model.png')
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1],shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, :]
    return image

def print_100(X_train):
    generated_images=X_train[:100]

    print generated_images.shape
    image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    # 调试阶段不生成图片
    Image.fromarray(image.astype(np.uint8)).save("keras-dcgan-svhn/raw.png")



def train(BATCH_SIZE=100):
    #load svhn 数据量太大了 使用其测试集来训练
    X_train = scipy.io.loadmat('svhn_32x32.mat', variable_names='X').get('X')
    Y_train = scipy.io.loadmat('svhn_32x32.mat', variable_names='y').get('y')

    print(X_train.shape, Y_train.shape)
    Y_train[Y_train == 10] = 0
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    #X_train = X_train[:, :, :, None]
    #(32, 32, 3,26032)
    X_train = np.swapaxes(X_train, 0, 3)
    #X_train = np.swapaxes(X_train, 1, 2)
    # 26032, 32, 3,32)
    X_train = np.swapaxes(X_train, 2, 3)
    # 26032, 32, 32,3)
    #调整前后关系  否则图像是横竖是反的
    X_train = np.swapaxes(X_train, 1, 2)

    #X_train = X_train.reshape(26032,32,32,3)

    print(X_train.shape, Y_train.shape)


    #打印前100个图案
    print_100(X_train)


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
        BATCH_COUNT = int(X_train.shape[0] / BATCH_SIZE)
        print("Epoch is", epoch)
        print("Number of batches",BATCH_COUNT )

        progress_bar = Progbar(target=BATCH_COUNT)
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 100 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                #调试阶段不生成图片
                Image.fromarray(image.astype(np.uint8)).save("keras-dcgan-svhn/"+str(epoch)+"_"+str(index)+".png")

            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            #print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            #print("batch %d g_loss : %f" % (index, g_loss))
            progress_bar.update(index,values=[("d_loss",d_loss),("g_loss",g_loss),("epoch",epoch+1)])


if __name__ == "__main__":
    train(BATCH_SIZE=100)
