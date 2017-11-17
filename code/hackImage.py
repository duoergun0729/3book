# -*- coding: utf-8 -*-

import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
from PIL import Image


#演示梯度下降求解的过程  损失函数为 y=x2+2
def demo1():
    import random
    a=0.1
    x=random.randint(1,10)
    y = x * x + 2
    index=1
    while index < 100 and abs(y-2) > 0.01 :
        y=x*x+2
        print "batch={} x={} y={}".format(index,x,y)
        x=x-2*x*a
        index+=1

#演示使用现成的模型进行判断
def demo2():
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    model = inception_v3.InceptionV3()
    img = image.load_img("pig.jpg", target_size=(299, 299))
    original_image = image.img_to_array(img)

    original_image /= 255.
    original_image -= 0.5
    original_image *= 2.

    original_image = np.expand_dims(original_image, axis=0)
    preds = model.predict(original_image)
    print('Predicted:', decode_predictions(preds, top=3)[0])

#demo2()
#demo()
#演示使用现成的模型进行判断
def demo3():
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(weights='imagenet')

    img_path = 'hacked-pig-image.png'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])

#demo3()

def demo4():
    model = inception_v3.InceptionV3()

    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    object_type_to_fake = 859


    img = image.load_img("pig.jpg", target_size=(299, 299))
    original_image = image.img_to_array(img)

    original_image /= 255.
    original_image -= 0.5
    original_image *= 2.

    original_image = np.expand_dims(original_image, axis=0)


    max_change_above = original_image + 0.01
    max_change_below = original_image - 0.01


    hacked_image = np.copy(original_image)


    learning_rate = 0.1


    cost_function = model_output_layer[0, object_type_to_fake]


    gradient_function = K.gradients(cost_function, model_input_layer)[0]


    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

    cost = 0.0

    index=1
    while cost < 0.60:

        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])


        hacked_image += gradients * learning_rate
        #print gradients

        hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
        hacked_image = np.clip(hacked_image, -1.0, 1.0)

        print("batch:{} Cost: {:.8}%".format(index,cost * 100))
        index+=1

    img = hacked_image[0]
    img /= 2.
    img += 0.5
    img *= 255.

    im = Image.fromarray(img.astype(np.uint8))
    im.save("hacked-pig-image.png")


def demo5():
    model = inception_v3.InceptionV3()

    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    object_type_to_fake = 859


    img = image.load_img("pig.jpg", target_size=(299, 299))
    original_image = image.img_to_array(img)

    original_image /= 255.
    original_image -= 0.5
    original_image *= 2.

    original_image = np.expand_dims(original_image, axis=0)


    max_change_above = original_image + 0.01
    max_change_below = original_image - 0.01


    hacked_image = np.copy(original_image)





    cost_function = model_output_layer[0, object_type_to_fake]


    gradient_function = K.gradients(cost_function, model_input_layer)[0]


    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

    cost = 0.0

    index=1

    learning_rate = 0.3


    e=0.007



    while cost < 0.99:

        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

        #fast gradient sign method
        #EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
        #hacked_image += gradients * learning_rate
        n=np.sign(gradients)
        hacked_image +=n*e
        #print gradients

        hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
        hacked_image = np.clip(hacked_image, -1.0, 1.0)

        print("batch:{} Cost: {:.8}%".format(index,cost * 100))
        index+=1

    img = hacked_image[0]
    img /= 2.
    img += 0.5
    img *= 255.

    im = Image.fromarray(img.astype(np.uint8))
    im.save("hacked-pig-image.png")

if __name__ == '__main__':
    demo5()