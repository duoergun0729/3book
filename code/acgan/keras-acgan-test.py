# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import defaultdict
import cPickle as pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout,add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
from keras.utils import plot_model

np.random.seed(0)

K.set_image_dim_ordering('th')


def build_generator(latent_size):

    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * 7 * 7, activation='relu'))
    cnn.add(Reshape((128, 7, 7)))


    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(256, (5, 5), padding="same", kernel_initializer="glorot_normal", activation="relu"))


    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, (5, 5), padding="same", kernel_initializer="glorot_normal", activation="relu"))


    cnn.add(Conv2D(1, (2, 2), padding="same", kernel_initializer="glorot_normal", activation="tanh"))


    latent = Input(shape=(latent_size, ))


    image_class = Input(shape=(1,), dtype='int32')


    cls = Flatten()(Embedding(10, 100, embeddings_initializer="glorot_normal")(image_class))

    #h = merge([latent, cls], mode='mul')
    h=add([latent, cls])

    fake_image = cnn(h)


    return Model(inputs=[latent, image_class], outputs=[fake_image])


def build_discriminator():

    cnn = Sequential()

    cnn.add(Conv2D(32, (3, 3), padding="same", strides=(2, 2), input_shape=(1, 28, 28) ))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, (3, 3), padding="same", strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, (3, 3), padding="same", strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 28, 28))

    features = cnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(inputs=[image], outputs=[fake, aux])

if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 50
    batch_size = 100
    latent_size = 100

    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    #discriminator.summary()

    #plot_model(discriminator, show_shapes=True, to_file='gan/acgan-discriminator.png')

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')
    generator.summary()
    #plot_model(generator, show_shapes=True, to_file='gan/acgan-generator.png')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(inputs=[latent, image_class], outputs=[fake, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    #plot_model(combined, show_shapes=True, to_file='gan/acgan-combined.png')

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=1)

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        #
        # 加载模型
        generator.load_weights(
            'params_generator_epoch_{0:03d}.hdf5'.format(epoch))
        discriminator.load_weights(
            'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch))


        print('\nTesting for epoch {}:'.format(epoch + 1))


        # generate some digits to display
        noise = np.random.uniform(-1, 1, (100, latent_size))

        sampled_labels = np.array([
            [i] * 10 for i in range(10)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            'plot_epoch_{0:03d}_generated.png'.format(epoch))
