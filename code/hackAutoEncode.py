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
from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Flatten
from keras.optimizers import SGD,RMSprop,adam
from keras.datasets import mnist
import keras
import numpy as np
from PIL import Image
import argparse
import math
import keras.backend as K
from keras.utils.generic_utils import Progbar
import os
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import plot_model




def trainCNN():
    batch_size = 128
    num_classes = 10
    epochs = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.summary()
    plot_model(model, show_shapes=True, to_file='hackImage/keras-cnn.png')

    #保证只有第一次调用的时候会训练参数
    if os.path.exists('hackImage/keras-cnn.h5'):
        model.load_weights('hackImage/keras-cnn.h5')
    else:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.save_weights("hackImage/keras-cnn.h5")

    return model


#获取100个非0样本
def getDataFromMnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #print x_train[0]
    #原有范围在0-255转换到 0-1
    #x_train = (x_train.astype(np.float32) - 127.5)/127.5
    #原有范围在0-255转换调整到-1和1之间
    x_train = x_train.astype(np.float32)/255.0
    x_train-=0.5
    x_train*=2.0


    x_train = x_train[:, :, :, None]
    x_test = x_test[:, :, :, None]

    #print type(y_train)
    index=np.where(y_train!=0)
    print "Raw train data shape:{}".format(x_train.shape)
    x_train=x_train[index]
    print "All 1-9 train data shape:{}".format(x_train.shape)
    x_train=x_train[-100:]
    print "Selected 100 1-9 train data shape:{}".format(x_train.shape)
    #print index
    #print type(index)
    #print y_train
    #print x_train
    return x_train


def get_images(generated_images):
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


def hackAll():
    raw_images=getDataFromMnist()
    generator_images=np.copy(raw_images)
    image=get_images(generator_images)
    image = (image/2.0+0.5)*255.0
    Image.fromarray(image.astype(np.uint8)).save("hackImage/100mnist-raw.png")

    image=255.0-image
    Image.fromarray(image.astype(np.uint8)).save("hackImage/100mnist-raw-w.png")

    cnn=trainCNN()
    #都伪装成0
    object_type_to_fake=0

    model_input_layer = cnn.layers[0].input
    model_output_layer = cnn.layers[-1].output

    cost_function = model_output_layer[0, object_type_to_fake]
    gradient_function = K.gradients(cost_function, model_input_layer)[0]
    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                    [cost_function, gradient_function])
    cost = 0.0
    #learning_rate = 0.3
    #e = 0.007
    e = 0.007

    progress_bar = Progbar(target=100)
    for index in range(100):
        #print "\nHack image:{}".format(index)
        progress_bar.update(index)

        mnist_image_raw=generator_images[index]
        mnist_image_hacked = np.copy(mnist_image_raw)
        #print mnist_image_hacked

        mnist_image_hacked = np.expand_dims(mnist_image_hacked, axis=0)
        #x = np.expand_dims(mnist_image_hacked, axis=0)
        preds = cnn.predict(mnist_image_hacked)
        #print "preds:{} mnist:{} likehood:{}\n".format(preds,np.argmax(preds),np.amax(preds))
        #调整的极限 彩色图片
        #max_change_above = mnist_image_raw + 0.01
        #max_change_below = mnist_image_raw - 0.01
        #调整的极限 灰度图片
        max_change_above = mnist_image_raw + 1.0
        max_change_below = mnist_image_raw - 1.0

        i=0
        cost=0
        while cost < 0.80:
            cost, gradients = grab_cost_and_gradients_from_model([mnist_image_hacked, 0])
            #print cost

            # fast gradient sign method
            # EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
            # hacked_image += gradients * learning_rate
            n = np.sign(gradients)
            mnist_image_hacked += n * e

            mnist_image_hacked = np.clip(mnist_image_hacked, max_change_below, max_change_above)
            mnist_image_hacked = np.clip(mnist_image_hacked, -1.0, 1.0)

            #print("batch:{} Cost: {:.8}%".format(i, cost * 100))
            #progress_bar.update(index,values=[('cost',cost),('batch',i)],force=True)
            i += 1

        #覆盖原有图片
        generator_images[index]=mnist_image_hacked

    #保存图片
    image=get_images(generator_images)
    image = (image/2.0+0.5)*255.0
    Image.fromarray(image.astype(np.uint8)).save("hackImage/100mnist-hacked.png")

    image=255.0-image

    Image.fromarray(image.astype(np.uint8)).save("hackImage/100mnist-hacked-w.png")


    #灰度图像里面黑是0 白是255 可以把中间状态的处理下
    image[image>127]=255
    Image.fromarray(image.astype(np.uint8)).save("hackImage/100mnist-hacked-w-good.png")

def trainAutoEncode():
    batch_size = 128
    epochs = 10

    h5file="hackAutoEncode/autoEncode.h5"

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    input_shape = (28*28,)

    #图像转换到-1到1之间
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train-=0.5
    x_train*=2.0
    x_test-=0.5
    x_test*=2.0

    print x_test

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #增加高斯噪声 均值为0 标准差为1
    x_train_nosiy = x_train + 0.3 * np.random.normal(loc=0., scale=1., size=x_train.shape)
    x_test_nosiy = x_test + 0.3 * np.random.normal(loc=0, scale=1, size=x_test.shape)
    x_train_nosiy = np.clip(x_train_nosiy, -1., 1.)
    x_test_nosiy = np.clip(x_test_nosiy, -1., 1.)
    print(x_train_nosiy.shape, x_test_nosiy.shape)

    input_img = Input(shape=input_shape)
    encoded = Dense(500, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    model = Model(inputs=[input_img], outputs=[decoded])


    model.compile(loss='binary_crossentropy',optimizer='adam')

    model.summary()
    plot_model(model, show_shapes=True, to_file='hackAutoEncode/keras-ae.png')

    #保证只有第一次调用的时候会训练参数
    if os.path.exists(h5file):
        model.load_weights(h5file)
    else:
        model.fit(x_train_nosiy, x_train, epochs=epochs, batch_size=batch_size,
                  verbose=1, validation_data=(x_test, x_test))

        model.save_weights(h5file)

    return model


def hackAutoEncode():
    raw_images=getDataFromMnist()
    generator_images=np.copy(raw_images)
    image=get_images(generator_images)
    image = (image/2.0+0.5)*255.0
    Image.fromarray(image.astype(np.uint8)).save("hackAutoEncode/100mnist-raw.png")
    image=255.0-image
    Image.fromarray(image.astype(np.uint8)).save("hackAutoEncode/100mnist-raw-w.png")

    model = trainAutoEncode()

    progress_bar = Progbar(target=100)
    for index in range(100):
        progress_bar.update(index)
        mnist_image_raw = generator_images[index]
        mnist_image_hacked = np.copy(mnist_image_raw)
        mnist_image_hacked=mnist_image_hacked.reshape(28*28)
        mnist_image_hacked = np.expand_dims(mnist_image_hacked, axis=0)
        #print mnist_image_hacked.shape
        preds = model.predict(mnist_image_hacked)
        mnist_image_hacked=preds
        #print preds

        #覆盖原有图片
        mnist_image_hacked=mnist_image_hacked.reshape(28,28,1)
        generator_images[index]=mnist_image_hacked

    #保存图片
    image=get_images(generator_images)
    image = (image/2.0+0.5)*255.0
    Image.fromarray(image.astype(np.uint8)).save("hackAutoEncode/100mnist-hacked.png")

    image=255.0-image

    Image.fromarray(image.astype(np.uint8)).save("hackAutoEncode/100mnist-hacked-w.png")


    #灰度图像里面黑是0 白是255 可以把中间状态的处理下
    image[image>127]=255
    Image.fromarray(image.astype(np.uint8)).save("hackAutoEncode/100mnist-hacked-w-good.png")

if __name__ == "__main__":
    #trainCNN()
    #hackAll()
    hackAutoEncode()







