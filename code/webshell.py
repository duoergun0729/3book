import time
import md5
from selenium import webdriver
import os
import re
import sys
import urllib2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import pickle
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

# data_file="../data/url/url.txt"
data_file = "../data/url/webshell.txt"
run_dir = "../data/url/pro"
pkl_file="xgboost.pkl"
pkl_vocabulary="vocabulary.pkl"
pkl_tfidf="tfidf.pkl"


max_features = 20000


def do_url_files(file):
    with open(file) as f:
        for line in f:
            line = line.strip('\n')
            print "Screenshot %s" % (line)
            m1 = md5.new()
            m1.update(line)
            screenshot_file = "../data/url/screenshot/" + m1.hexdigest() + ".png"
            # print screenshot_file
            try:
                open_url(line, screenshot_file)
            except:
                print "Fail to screenshot"


def open_url(url, photo_file):
    browser = webdriver.Chrome()
    browser.set_window_size(1000, 800)
    browser.get(url)
    time.sleep(5)
    browser.save_screenshot(photo_file)
    browser.quit()


def get_url_from_dir(path, old, new):
    for r, d, files in os.walk(path):
        for file in files:
            if file.endswith('.php'):
                file_path = os.path.join(r, file)
                # print "Load %s" % file_path
                file_path = file_path.replace(old, new)
                print "%s" % file_path


def get_hao123():
    response = urllib2.urlopen('https://www.hao123.com/')
    html = response.read()
    # print html
    url_list = re.findall(r'href="(http://\S+)"', html)
    print url_list
    for url in url_list:
        print url


# get_hao123()


# get_url_from_dir("../../2book/data/webshell/webshell/","../../2book/data/webshell/","http://127.0.0.1:8080/")


# do_url_files(data_file)

def get_token_from_file(lines):

    #print lines
    #+re.findall(r'>([^>]+)</a>', lines, re.I)\
    #+re.findall(r'>([^>]+)</p>', lines, re.I)
    #+re.findall(r'>([^>]+)</br>', lines, re.I) \
    #+re.findall(r'>([^>]+)</b>', lines, re.I)

    #x = re.findall(r'<title>([^>]+)</title>', lines,re.I)

    #good
    #x = re.findall(r'\b\w+\b', lines, re.I)
    x = re.findall(r'"><b>([^>]+)</b>', lines, re.I)+\
        re.findall(r'<center>"([^<]+)"<', lines, re.I)+\
        re.findall(r'value="([^"]+)"\s+id', lines, re.I)+\
        re.findall(r'<title>([^>]+)</title>', lines, re.I)+\
        re.findall(r'>([^>]+)</font>', lines, re.I)+\
        re.findall(r'<b>([^>]+)<b>', lines, re.I)

    #print x
    return " ".join(x)


def load_one_file(filename):
    x = ""
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            x += line
    return x


def load_files_from_dir(rootdir):
    x = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v = load_one_file(path)
            v = get_token_from_file(v)
            print "path:%s" % path
            print v
            x.append(v)
    return x


def load_all_files():
    ham = []
    spam = []

    path = "../data/url/webshell"
    print "Load %s" % path
    spam = load_files_from_dir(path)
    #print spam

    path = "../data/url/normal"
    #path = "../data/url/webshell"
    print "Load %s" % path
    ham = load_files_from_dir(path)

    return ham, spam


def get_features_by_wordbag():
    ham, spam = load_all_files()
    #print spam
    x = ham + spam
    y = [0] * len(ham) + [1] * len(spam)
    vectorizer = CountVectorizer(
        #token_pattern=r'\s\b\w+\b\s',
        token_pattern=r'\b\w+\b',
        decode_error='ignore',
        strip_accents='ascii',
        max_features=max_features,
        stop_words='english',
        max_df=1.0,
        min_df=1)
    print vectorizer
    #vocabulary_=vectorizer.vocabulary_
    x = vectorizer.fit_transform(x)
    x = x.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    x = transformer.fit_transform(x)
    x = x.toarray()
    print "Black %d White %d" % (len(spam), len(ham))

    joblib.dump(vectorizer, pkl_vocabulary)
    joblib.dump(transformer, pkl_tfidf)
    return x, y


def do_metrics(y_test, y_pred):
    print "metrics.accuracy_score:"
    print metrics.accuracy_score(y_test, y_pred)
    print "metrics.confusion_matrix:"
    print metrics.confusion_matrix(y_test, y_pred)
    print "metrics.precision_score:"
    print metrics.precision_score(y_test, y_pred)
    print "metrics.recall_score:"
    print metrics.recall_score(y_test, y_pred)
    print "metrics.f1_score:"
    print metrics.f1_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print "metrics.auc:"
    print metrics.auc(fpr, tpr)


def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print "NB and wordbag"
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    do_metrics(y_test, y_pred)

def do_mlp(x_train, x_test, y_train, y_test):
    print "mlp"
    #mlp
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)


    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    do_metrics(y_test,y_pred)

def do_xgboost(x_train, x_test, y_train, y_test):
    print "xgboost"
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    do_metrics(y_test, y_pred)
    joblib.dump(xgb_model, pkl_file)

def do_RandomForest(x_train, x_test, y_train, y_test):
    print "RandomForest"
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    do_metrics(y_test, y_pred)


def run_xgboost_from_pkl():
    n=0
    xgb_model = joblib.load(pkl_file)
    print "Load files from %s" % run_dir


    vectorizer=joblib.load(pkl_vocabulary)
    transformer = joblib.load(pkl_tfidf)

    rootdir=run_dir
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            vv = load_one_file(path)
            vv = get_token_from_file(vv)
            #print "Path:%s" % path
            #print vv
            v=[]
            v.append(vv)
            v = vectorizer.transform(v)
            v = v.toarray()
            v = transformer.transform(v)
            v = v.toarray()
            pred = xgb_model.predict(v)
            if pred[0] == 1:
                n=n+1
                print "Path:%s" % path
                print vv
    print n



if __name__ == "__main__":
    x, y = get_features_by_wordbag()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    #do_nb_wordbag(x_train, x_test, y_train, y_test)
    #do_mlp(x_train, x_test, y_train, y_test)
    do_xgboost(x_train, x_test, y_train, y_test)
    #run_xgboost_from_pkl()
    #do_RandomForest(x_train, x_test, y_train, y_test)
