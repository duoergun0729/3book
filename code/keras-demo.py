# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model

def func1():
    print "func1"
    model = Sequential()
    model.add(Dense(32, input_shape=(784,)))
    model.add(Activation('relu'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    plot_model(model, to_file='keras-demo1.png')
    plot_model(model,show_shapes=True, to_file='keras-demo2.png')



if __name__ == '__main__':
    func1()