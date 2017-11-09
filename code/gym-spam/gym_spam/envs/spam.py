#-*- coding:utf-8 –*-

import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import os
#from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from features import Features


max_features=15000
mode_file="spam_mlp.pkl"
vocabulary_file="spam_vocabulary.pkl"

local_model_threshold = 0.6


def load_one_file(filename):
    x=""
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            line = line.strip('\r')
            x+=line
    return x

def load_files_from_dir(rootdir):
    x=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v=load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    ham=[]
    spam=[]
    for i in range(1,7):
        path="../../data/mail/enron%d/ham/" % i
        print "Load %s" % path
        ham+=load_files_from_dir(path)
        path="../../data/mail/enron%d/spam/" % i
        print "Load %s" % path
        spam+=load_files_from_dir(path)
    return ham,spam

def load_all_spam():
    spam=[]
    for i in range(1,3):
        path="../../data/mail/enron%d/spam/" % i
        print "Load %s" % path
        spam+=load_files_from_dir(path)

    # 划分训练和测试集合
    samples_train, samples_test = train_test_split(spam, test_size=100)
    return samples_train, samples_test


def get_features_by_wordbag():
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vectorizer=None

    if os.path.exists(vocabulary_file):
        vocabulary=joblib.load(vocabulary_file)
        vectorizer = CountVectorizer(
                                     decode_error='ignore',
                                     vocabulary=vocabulary,
                                     strip_accents='ascii',
                                     max_features=max_features,
                                     stop_words='english',
                                     max_df=1.0,
                                     min_df=1 )
    else:
        vectorizer = CountVectorizer(
                                     decode_error='ignore',
                                     strip_accents='ascii',
                                     max_features=max_features,
                                     stop_words='english',
                                     max_df=1.0,
                                     min_df=1 )

    print vectorizer
    x=vectorizer.fit_transform(x)
    x=x.toarray()

    if not os.path.exists(vocabulary_file):
        vocabulary_=vectorizer.vocabulary_
        joblib.dump(vocabulary_,vocabulary_file)

    return x,y


def do_mlp_wordbag(x_train, x_test, y_train, y_test):
    print "mlp and wordbag"

    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print  clf
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print "accuracy_score:"
    print metrics.accuracy_score(y_test, y_pred)
    print "recall_score:"
    print metrics.recall_score(y_test, y_pred)
    print "precision_score:"
    print metrics.precision_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

    joblib.dump(clf,mode_file)

def train_mlp_spam():
    x,y=get_features_by_wordbag()

    train_X, test_X, train_y, test_y = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=66)

    do_mlp_wordbag(train_X, test_X, train_y, test_y)

class Spam_Check(object):
    def __init__(self):
        self.name="Spam_Check"
        self.clf=joblib.load(mode_file)
        #self.features_extract=Features()

    def check_spam(self,featurevectors):
        #[[ 0.96085352  0.03914648]]  返回的是垃圾邮件的概率
        y_pred = self.clf.predict_proba([featurevectors])[0,-1]
        #大于阈值的判断为垃圾邮件
        label = float(y_pred >= local_model_threshold)
        return label




if __name__ == '__main__':

    train_mlp_spam()


    spam_Check=Spam_Check()
    features_extract = Features(vocabulary_file)
    featurevectors=features_extract.extract("thank you ,your email address was obtained from a purchased list ,"
                "reference # 2020 mid = 3300 . if you wish to unsubscribe")
    spam_Check.check_spam(featurevectors)

    samples_train, samples_test=load_all_spam()

    sum=0
    success=0

    for sample in samples_test:
        sum+=1
        featurevectors = features_extract.extract(sample)
        label=spam_Check.check_spam(featurevectors)
        print label
        if label == 1.0:
            success+=1
        print "sum={} success={} success rate={}".format(sum, success,float(success)/sum)

    #print "sum={} success={}".format(sum,success)
