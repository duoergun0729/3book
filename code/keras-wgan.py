# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD,RMSprop,adam
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import keras.backend as K
from keras.utils.generic_utils import Progbar
#from keras.utils import plot_model


#wgan的改进
#损失函数使用w值定义
# 训练d之后 修正参数 wgan的精髓之一

#定义w距离为损失函数
def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    #plot_model(model, show_shapes=True, to_file='wgan/keras-wgan-generator_model.png')
    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (3, 3),strides=(2, 2),padding='same',
            input_shape=(28, 28, 1))
            )
    #model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(32, (3, 3),strides=(2, 2),padding='same'))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(16, (3, 3),strides=(2, 2),padding='same'))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(1, (3, 3),padding='same'))

    model.add(GlobalAveragePooling2D())

    #plot_model(model, show_shapes=True, to_file='wgan/keras-wgan-discriminator_model.png')
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)

    #plot_model(model, show_shapes=True, to_file='wgan/keras-dcgan-gan_model.png')
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
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    #d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    #g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    d_optim = RMSprop(lr=5E-5)
    g_optim = RMSprop(lr=5E-5)

    c_lower = -0.1
    c_upper = 0.1




    #g的损失函数使用mse
    #g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.compile(loss='mse', optimizer=g_optim)
    #gan的损失函数使用wasserstein
    #d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d_on_g.compile(loss=wasserstein, optimizer=g_optim)
    d.trainable = True
    #d的损失函数使用wasserstein
    d.compile(loss=wasserstein, optimizer=d_optim)

    for epoch in range(100):
        print "Epoch is {}/100".format(epoch)
        nb_batches=int(X_train.shape[0]/BATCH_SIZE)
        print("Number of batches", nb_batches)

        progress_bar = Progbar(target=nb_batches)
        for index in range(nb_batches):
            progress_bar.update(index)


            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 100 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                #调试阶段不生成图片
                Image.fromarray(image.astype(np.uint8)).save('wgan/'+str(epoch)+"_"+str(index)+".png")

            X = np.concatenate((image_batch, generated_images))
            y = [-1] * BATCH_SIZE + [1] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)

            # 训练d之后 修正参数 wgan的精髓之一
            for l in d.layers:
                weights = l.get_weights()
                #print weights
                weights = [np.clip(w, c_lower, c_upper) for w in weights]
                l.set_weights(weights)


            #print("epoch %d/100 batch %d d_loss : %f" % (epoch+1, index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [-1] * BATCH_SIZE)
            d.trainable = True
            #print("epoch %d/100 batch %d g_loss : %f" % (epoch+1,index, g_loss))
            if index % 10 == 9:
                g.save_weights('wgan/wgan_generator', True)
                d.save_weights('wgan/wgan_discriminator', True)



def wgan():
    train(BATCH_SIZE=100)


if __name__ == "__main__":
    wgan()
