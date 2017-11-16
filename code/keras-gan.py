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


mu, sigma=(0,1)

#真实样本满足正态分布 平均值维0 方差为1 样本维度200
def x_sample(size=200,batch_size=32):
    x=[]
    for _ in range(batch_size):
        x.append(np.random.normal(mu, sigma, size))
    return np.array(x)

#噪声样本 噪声维度维200
def z_sample(size=200,batch_size=32):
    z=[]
    for _ in range(batch_size):
        z.append(np.random.uniform(-1, 1, size))
    return np.array(z)


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=200, units=256))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('sigmoid'))
    plot_model(model, show_shapes=True, to_file='gan/keras-gan-generator_model.png')
    return model


def discriminator_model():
    model = Sequential()

    model.add(Reshape((200,), input_shape=(200,)))
    model.add(Dense(units=256))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    plot_model(model, show_shapes=True, to_file='gan/keras-gan-discriminator_model.png')
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    plot_model(model, show_shapes=True, to_file='gan/keras-gan-gan_model.png')
    return model

def show_image(s):
    count, bins, ignored = plt.hist(s, 5, normed=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()

def save_image(s,filename):
    count, bins, ignored = plt.hist(s, bins=20, normed=True,facecolor='w',edgecolor='b')
    y = MLA.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'g--', linewidth=2)
    #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.savefig(filename)



def show_init():
    x=x_sample(batch_size=1)[0]
    save_image(x,"gan/x-0.png")
    z=z_sample(batch_size=1)[0]
    save_image(z, "gan/z-0.png")


def save_loss(d_loss_list,g_loss_list):

    plt.subplot(2, 1, 1)  # 面板设置成2行1列，并取第一个（顺时针编号）
    plt.plot(d_loss_list, 'yo-')  # 画图，染色
    #plt.title('A tale of 2 subplots')
    plt.ylabel('d_loss')

    plt.subplot(2, 1, 2)  # 面板设置成2行1列，并取第二个（顺时针编号）
    plt.plot(g_loss_list,'r.-')  # 画图，染色
    #plt.xlabel('time (s)')
    plt.ylabel('g_loss')


    plt.savefig("gan/loss.png")

if __name__ == '__main__':

    show_init()
    d_loss_list=[]
    g_loss_list = []


    batch_size=128
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(500):
        print("Epoch is", epoch)
        noise=z_sample(batch_size=batch_size)
        image_batch=x_sample(batch_size=batch_size)
        generated_images = g.predict(noise, verbose=0)
        x= np.concatenate((image_batch, generated_images))
        y=[1]*batch_size+[0]*batch_size
        d_loss = d.train_on_batch(x, y)
        print("d_loss : %f" % (d_loss))
        noise = z_sample(batch_size=batch_size)
        d.trainable = False
        g_loss = d_on_g.train_on_batch(noise, [1]*batch_size)
        d.trainable = True
        print("g_loss : %f" % (g_loss))
        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss)

        if epoch % 100 == 1:
            # 测试阶段
            noise = z_sample(batch_size=1)
            generated_images = g.predict(noise, verbose=0)
            # print generated_images
            save_image(generated_images[0], "gan/z-{}.png".format(epoch))

    save_loss(d_loss_list, g_loss_list)



