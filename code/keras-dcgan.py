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
from keras.utils import plot_model


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

    plot_model(model, show_shapes=True, to_file='keras-dcgan-generator_model.png')
    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
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

    plot_model(model, show_shapes=True, to_file='keras-dcgan-discriminator_model.png')
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)

    plot_model(model, show_shapes=True, to_file='keras-dcgan-gan_model.png')
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
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                #调试阶段不生成图片
                #Image.fromarray(image.astype(np.uint8)).save(str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('dcgan_generator', True)
                d.save_weights('dcgan_discriminator', True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save("generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


def demo1():
    import matplotlib.pyplot as plt
    import numpy as np

    s = np.random.uniform(0, 1, 1200)
    count, bins, ignored = plt.hist(s, 12, normed=True)

    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()


def demo2():
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as MLA

    mu, sigma = 10, 10
    x = mu + sigma * np.random.randn(5000)

    n, bins, patches = plt.hist(x, 20, normed=1, facecolor='blue', alpha=0.8)

    y = MLA.normpdf(bins, mu, sigma)

    l = plt.plot(bins, y, 'g--', linewidth=3)

    plt.xlabel('samples')
    plt.ylabel('p')
    plt.title(r'$Normal\ pdf\ m=10,\ \sigma=10$')
    plt.axis([-30, 50, 0, 0.042])
    plt.grid(True)
    plt.show()


def maidou_gan():
    def gan_generator_model():
        model = Sequential()
        model.add(Dense(input_dim=200, units=256))
        model.add(Activation('relu'))
        model.add(Dense(28*28*1))
        model.add(Activation('sigmoid'))

        model.add(Reshape((28, 28, 1), input_shape=(28*28*1,)))

        plot_model(model, show_shapes=True, to_file='keras-gan-generator_model.png')
        return model

    def gan_discriminator_model():
        model = Sequential()

        model.add(Reshape((28*28*1,), input_shape=(28, 28, 1)))
        model.add(Dense(units=256))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        plot_model(model, show_shapes=True, to_file='keras-gan-discriminator_model.png')
        return model

    def gan_generator_containing_discriminator(g, d):
        model = Sequential()
        model.add(g)
        d.trainable = False
        model.add(d)

        plot_model(model, show_shapes=True, to_file='keras-gan-gan_model.png')
        return model

    def gan_combine_images(generated_images):

        #print generated_images.shape[0]

        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
        shape = generated_images.shape[1:3]
        image = np.zeros((height * shape[0], width * shape[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index / width)
            j = index % width
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
                img[:, :, 0]
        return image

    def gan_train(BATCH_SIZE):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        #把像素点转化成小数
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = X_train[:, :, :, None]
        X_test = X_test[:, :, :, None]

        print X_train.shape
        # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
        d = gan_discriminator_model()
        g = gan_generator_model()
        d_on_g = gan_generator_containing_discriminator(g, d)
        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g.compile(loss='binary_crossentropy', optimizer='adam')
        d_on_g.compile(loss='binary_crossentropy', optimizer='adam')
        d.trainable = True
        d.compile(loss='binary_crossentropy', optimizer='adam')
        for epoch in range(100):
            print("Epoch is", epoch)
            print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
            for index in range(int(X_train.shape[0] / BATCH_SIZE)):
                noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 200))
                image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                generated_images = g.predict(noise, verbose=0)
                if index % 200 == 0:
                    image = gan_combine_images(generated_images)
                    image = image * 127.5 + 127.5
                    # 调试阶段不生成图片
                    Image.fromarray(image.astype(np.uint8)).save("gan/"+str(epoch)+"_"+str(index)+".png")
                X = np.concatenate((image_batch, generated_images))
                y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
                d_loss = d.train_on_batch(X, y)
                print("batch %d d_loss : %f" % (index, d_loss))
                noise = np.random.uniform(-1, 1, (BATCH_SIZE, 200))
                d.trainable = False
                g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
                d.trainable = True
                print("batch %d g_loss : %f" % (index, g_loss))
                if index % 10 == 9:
                    g.save_weights('gan_generator', True)
                    d.save_weights('gan_discriminator', True)

    print "maidou"
    gan_train(BATCH_SIZE=128)

def maidou_dcgan():
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)

if __name__ == "__main__":
    maidou_gan()
