import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import metrics

batch_size = 128
num_classes = 10
epochs = 4

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#print score


y_pred = np.argmax(model.predict(x_test),axis=1)
y_true = np.argmax(y_test, axis = 1)
cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
print cm

print('metrics.accuracy_score:', metrics.accuracy_score(y_true,y_pred))
print('metrics.precision_score:', metrics.precision_score(y_true,y_pred,average="micro"))
print('metrics.recall_score:', metrics.recall_score(y_true,y_pred,average="micro"))
#print('metrics.roc_auc_score:', metrics.roc_auc_score(y_true,y_pred,average="micro"))




