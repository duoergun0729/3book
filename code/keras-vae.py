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
from keras.layers import Dense, Dropout, Flatten,Lambda
from keras.layers.core import Flatten
from keras.optimizers import SGD,RMSprop,adam
from keras.losses import binary_crossentropy
from keras.datasets import mnist
import keras
from keras import metrics
import numpy as np
from PIL import Image
import argparse
import math
import keras.backend as K
from keras.utils.generic_utils import Progbar
import os
from keras.layers import Dense, Input
from keras.models import Model
#from keras.utils import plot_model


batch_size = 200
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 20
epsilon_std = 1.0


#获取100个非0样本
def getDataFromMnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #print x_train[0]
    #原有范围在0-255转换到 0-1
    #x_train = (x_train.astype(np.float32) - 127.5)/127.5
    #原有范围在0-255转换调整到-1和1之间
    #x_train = x_train.astype(np.float32)/255.0
    #x_train-=0.5
    #x_train*=2.0
    # 原有范围在0-255转换调整到0和1之间
    x_train = x_train.astype(np.float32) / 255.0

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


#获取数字0的图案
def getZeroFromMnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #print x_train[0]
    #原有范围在0-255转换到 0-1
    #x_train = (x_train.astype(np.float32) - 127.5)/127.5
    #原有范围在0-255转换调整到-1和1之间
    #x_train = x_train.astype(np.float32)/255.0
    #x_train-=0.5
    #x_train*=2.0
    #原有范围在0-255转换调整到0和1之间
    x_train = x_train.astype(np.float32)/255.0



    x_train = x_train[:, :, :, None]
    x_test = x_test[:, :, :, None]

    index=np.where(y_train==0)
    print "Raw train data shape:{}".format(x_train.shape)
    x_train=x_train[index]
    print "All 0 train data shape:{}".format(x_train.shape)
    x_train=x_train[-1:]
    print "Selected 1 1-9 train data shape:{}".format(x_train.shape)

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



def trainAutoEncode():


    h5file="VAE/autoEncode.h5"

    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    #图像转换到0到1之间
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    #x_train-=0.5
    #x_train*=2.0
    #x_test-=0.5
    #x_test*=2.0

    #print x_test

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_std = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_std = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_std) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_std])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    def vae_loss(x, x_decoded_mean):
        encode_decode_loss=original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_std - K.square(z_mean) - K.exp(z_std), axis=-1)

        return  K.mean(kl_loss+encode_decode_loss)

    model = Model(x, x_decoded_mean)
    model.compile(optimizer='rmsprop', loss=vae_loss)



    #保证只有第一次调用的时候会训练参数
    if os.path.exists(h5file):
        model.load_weights(h5file)
    else:
        model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                  verbose=1, validation_data=(x_test, x_test))

        model.save_weights(h5file)


    #验证还原原始图像
    raw_images = getDataFromMnist()
    generator_images = np.copy(raw_images)

    #保存原始图片
    original_images=np.copy(raw_images)
    image = get_images(original_images)
    #image = (image / 2.0 + 0.5) * 255.0
    image = image * 255.0
    Image.fromarray(image.astype(np.uint8)).save("VAE/Original-b.png")
    image = 255.0 - image
    Image.fromarray(image.astype(np.uint8)).save("VAE/Original-w.png")

    #VAE还原
    generator_images = generator_images.reshape(100, 784)

    for index in range(100):
        mnist_image_raw = generator_images[index]
        mnist_image_hacked = np.copy(mnist_image_raw)
        mnist_image_hacked=mnist_image_hacked.reshape(784)
        mnist_image_hacked = np.expand_dims(mnist_image_hacked, axis=0)
        preds = model.predict(mnist_image_hacked)
        generator_images[index] = preds

    generator_images = generator_images.reshape(100, 28, 28, 1)
    image = get_images(generator_images)
    #image = (image / 2.0 + 0.5) * 255.0
    image = image * 255.0
    Image.fromarray(image.astype(np.uint8)).save("VAE/Original-Reconstruction-b.png")
    image = 255.0 - image
    Image.fromarray(image.astype(np.uint8)).save("VAE/Original-Reconstruction-w.png")

    #plot_model(model, show_shapes=True, to_file='VAE/vae_model.png')

    return model



def hackAutoEncode():
    raw_images=getDataFromMnist()
    generator_images=np.copy(raw_images)

    generator_images=generator_images.reshape(100,784)

    model = trainAutoEncode()

    # 都伪装成0
    object_type_to_fake = getZeroFromMnist()
    object_type_to_fake=object_type_to_fake.reshape(28*28)
    object_type_to_fake = np.expand_dims(object_type_to_fake, axis=0)
    #print object_type_to_fake

    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output
    #生成的图像与图案0之间的差为损失函数
    #cost_function = binary_crossentropy(y_pred,object_type_to_fake)
    cost_function = K.mean(K.binary_crossentropy(model_output_layer,object_type_to_fake))
    #cost_function=K.mean(K.square(model_output_layer-object_type_to_fake))
    gradient_function = K.gradients(cost_function, model_input_layer)[0]
    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                    [cost_function, gradient_function])
    cost = 0.0

    e = 0.007


    progress_bar = Progbar(target=100)
    for index in range(100):
        #print index
        progress_bar.update(index)
        mnist_image_raw = generator_images[index]
        mnist_image_hacked = np.copy(mnist_image_raw)
        mnist_image_hacked=mnist_image_hacked.reshape(28*28)
        mnist_image_hacked = np.expand_dims(mnist_image_hacked, axis=0)


        #print "mnist_image_hacked.shape:{}".format(mnist_image_hacked.shape)

        #调整的极限 灰度图片
        max_change_above = mnist_image_raw + 1.0
        max_change_below = mnist_image_raw - 1.0

        i=0
        cost=100
        """
        Epoch 18/20
60000/60000 [==============================] - 8s - loss: 153.7977 - val_loss: 153.8969
Epoch 19/20
60000/60000 [==============================] - 8s - loss: 153.6312 - val_loss: 153.5625
Epoch 20/20
60000/60000 [==============================] - 8s - loss: 153.7320 - val_loss: 156.0068
        """
        #print "\nmnist_image_hacked.shape:{}".format(mnist_image_hacked.shape)
        while cost > 156/784.0 and i < 500:
            #print "\nmnist_image_hacked.shape:{}".format(mnist_image_hacked.shape)
            cost, gradients = grab_cost_and_gradients_from_model([mnist_image_hacked, 0])
            #print cost

            # fast gradient sign method
            # EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
            # hacked_image += gradients * learning_rate
            n = np.sign(gradients)
            mnist_image_hacked -= n * e

            mnist_image_hacked = np.clip(mnist_image_hacked, max_change_below, max_change_above)
            #mnist_image_hacked = np.clip(mnist_image_hacked, -1.0, 1.0)
            #像素取值在0到1之间
            mnist_image_hacked = np.clip(mnist_image_hacked, 0.0, 1.0)

            #print("batch:{} Cost: {:.8}%".format(i, cost * 100))
            #progress_bar.update(index,values=[('cost',cost),('batch',i)],force=True)
            i += 1



        #覆盖原有图片
        #mnist_image_hacked=mnist_image_hacked.reshape(28,28,1)
        generator_images[index]=mnist_image_hacked

    autoEncode_images = np.copy(generator_images)
    generator_images=generator_images.reshape(100,28,28,1)



    #保存图片
    image=get_images(generator_images)
    #image = (image/2.0+0.5)*255.0
    image = image * 255.0
    Image.fromarray(image.astype(np.uint8)).save("VAE/AdversarialExamples-b.png")

    image=255.0-image

    Image.fromarray(image.astype(np.uint8)).save("VAE/AdversarialExamples-w.png")


    #灰度图像里面黑是0 白是255 可以把中间状态的处理下
    #image[image>127]=255
    #Image.fromarray(image.astype(np.uint8)).save("hackAutoEncode/100mnist-hacked-w-good.png")


    for index in range(100):
        mnist_image_raw = autoEncode_images[index]
        mnist_image_hacked = np.copy(mnist_image_raw)
        mnist_image_hacked=mnist_image_hacked.reshape(28*28)
        mnist_image_hacked = np.expand_dims(mnist_image_hacked, axis=0)
        preds = model.predict(mnist_image_hacked)

        autoEncode_images[index]=preds

    autoEncode_images = autoEncode_images.reshape(100, 28, 28, 1)
    image = get_images(autoEncode_images)
    #image = (image / 2.0 + 0.5) * 255.0
    image = image * 255.0
    Image.fromarray(image.astype(np.uint8)).save("VAE/ReconstructionAdversarialExamples-b.png")

    image=255.0-image

    Image.fromarray(image.astype(np.uint8)).save("VAE/ReconstructionAdversarialExamples-w.png")



if __name__ == "__main__":
    #trainCNN()
    #hackAll()
    hackAutoEncode()
    #trainAutoEncode()







