# -*- coding: utf-8 -*-
import os,sys,fnmatch
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import re

#pip install tensorflow-gpu keras h5py
# mkdir -p aisec/data/
#scp aisec.py aisec/data/tomcat.apache.org.tar.gz root@101.236.50.226:/data
html_dir="aisec/data/tomcat.apache.org/"
#html_dir="aisec/data/demo/"
model_file="model-aisec.h5"
epochs=10
max_files=1000

test_set=["lass=''><h2 id=''></h2><div clas",
          "lass=''><h id=''></h><div clas"
          ]


def search_file(pattern="*.txt", root=os.curdir):
    for path, dirs, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)

def load_html_files(root_path):
    file_suffix="*.html"
    return search_file(file_suffix, root_path)

#temperature 1为基准 越小约保守  越大越开放
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate(text, maxlen = 40,step = 1):
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))


    print char_indices

    # cut the text in semi-redundant sequences of maxlen characters

    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    # 避免过拟合
    model.add(LSTM(128, dropout=0.1,return_sequences=True,input_shape=(maxlen, len(chars))))

    model.add(LSTM(128, dropout=0.1, return_sequences=False))

    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    model.summary()


    """
    Temperature. We can also play with the temperature of the Softmax during sampling. Decreasing the temperature from 1
    to some lower number (e.g. 0.5) makes the RNN more confident, but also more conservative in its samples. Conversely,
    higher temperatures will give more diversity but at cost of more mistakes (e.g. spelling mistakes, etc). In particular,
    setting temperature very near zero will give the most likely thing that Paul Graham might say:

    """
    #相当于实现softmax
    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    if os.path.exists(model_file):
        model.load_weights(model_file)
        print()
    else:
        tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/TensorBoard', histogram_freq=0, write_graph=True,
                                                 write_images=True)
        model.fit(x, y,
                  batch_size=128,callbacks=[tbCallBack],
                  epochs=epochs)

        model.save_weights(model_file)

    # train the model, output generated text after each iteration
    for iteration in range(1, 2):
        print()
        print('-' * 50)
        print('Iteration', iteration)


        for diversity in [0.1, 1,1.8]:
            print()
            print('----- diversity:', diversity)

            for _ in range(6):
            #for sentence in test_set:
                #generated = ''
                start_index = random.randint(0, len(text) - maxlen - 1)
                sentence = text[start_index: start_index + maxlen]
                #sentence="<textarea>XXX"

                generated = ""
                generated += sentence
                #generated=""
                #便于区分种子很生成的内容
                generated += "||"
                print('\n')
                print('----- Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

                for i in range(16):
                    x_pred = np.zeros((1, maxlen, len(chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, char_indices[char]] = 1.

                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = sample(preds, diversity)
                    next_char = indices_char[next_index]

                    generated += next_char
                    sentence = sentence[1:] + next_char

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
"""
泛化处理
<a href="lists.html">泛化为<a href=X>

<em>not-yet-released</em>泛化为<em>X</em>

location.href.indexof('is-external=true') == -1泛化为location.href.indexof(X) == -1

document.getelementbyid("allclasses_navbar_bottom")泛化为document.getelementbyid(X)

<!-- ========= end of class data ========= --> 注释干掉

/* the following code is added by mdx_elementid.py
   it was originally lifted from http://subversion.apache.org/style/site.css */
/*
 * hide class=X, except when an enclosing heading
 * has the :hover property.
 */
注释干掉
"""
def clean_html_content(html_files):
    text = ""

    for i in html_files:
        one_html = open(i).read().lower()
        #忽略非ascii码
        #re.sub(r'[^\x00-\x7F]+',' ', text)
        one_html, _ = re.subn(r'[^\x00-\x7F]+', "", one_html, count=0, flags=re.M | re.S)

        #去除注释内容 需要多行匹配
        one_html, _ = re.subn(r'<!--.*?-->', "", one_html,count=0,flags=re.M|re.S)
        one_html, _ = re.subn(r'/\*.*?\*/', "", one_html, count=0, flags=re.M | re.S)

        one_html, _ = re.subn(r'\'[^\']+\'', "''", one_html, count=0, flags=re.M | re.S)
        one_html, _ = re.subn(r'>[^<>]+<', "><", one_html,count=0,flags=re.M|re.S)
        one_html, _ = re.subn(r'=\'[^\']+\'', "=''", one_html,count=0,flags=re.M|re.S)
        one_html, _ = re.subn(r'="[^"]+"', "=''", one_html,count=0,flags=re.M|re.S)
        one_html, _ = re.subn(r'"[^"]+"', "''", one_html,count=0,flags=re.M|re.S)


        #奇葩的存在
        #one_html, _ = re.subn(r'>[^<>]+>', ">X<", one_html)
        #one_html, _ = re.subn(r'=\s+"[^"]+"', "=''", one_html)
        #one_html, _ = re.subn(r'>[^<>]+<', "><", one_html)
        #


        text += one_html

    return text

if __name__ == '__main__':
    html_files_list=load_html_files(html_dir)
    html_files=[i for i in html_files_list]
    #性能考虑 仅处理指定个数的文件
    print('nb html_files:', len(html_files))
    html_files=html_files[:max_files]
    print('do with nb html_files:', len(html_files))

    clean_html=clean_html_content(html_files)

    #print clean_html

    generate(clean_html, maxlen=32, step=3)
    """
        text=""
        for i in html_files:
            html=open(i).read().lower()
            text+=html
        generate(text,maxlen = 40,step = 40)
    """

