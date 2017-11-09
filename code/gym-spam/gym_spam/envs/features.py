#-*- coding:utf-8 –*-

import numpy as np
#from spam import mode_file,vocabulary_file
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib



class Features(object):
    def __init__(self,vocabulary_file):
        self.dim = 0
        self.name="Spam_Features"
        self.dtype=np.float32
        self.vocabulary_file=vocabulary_file


    def extract(self,str):
        featurevectors=None
        #print str
        if os.path.exists(self.vocabulary_file):
            vocabulary = joblib.load(self.vocabulary_file)
            vectorizer = CountVectorizer(   decode_error='ignore',
                                            vocabulary=vocabulary,
                                            strip_accents='ascii',
                                            stop_words='english',
                                            max_df=1.0,
                                            min_df=1)
            #CountVectorizer产生的是稀疏矩阵 所以需要使用toarray()转换成了numpy结构
            featurevectors=vectorizer.transform([str]).toarray()

        return np.concatenate(featurevectors)

