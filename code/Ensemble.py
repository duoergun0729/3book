# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics


def func1():
    x, y = datasets.make_classification(n_samples=1000, n_features=100,n_redundant=0, random_state = 1)
    train_X, test_X, train_Y, test_Y = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=66)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_X, train_Y)
    pred_Y = knn.predict(test_X)

    print "accuracy_score:"
    print metrics.accuracy_score(test_Y, pred_Y)
    print "f1_score:"
    print metrics.f1_score(test_Y, pred_Y)
    print "recall_score:"
    print metrics.recall_score(test_Y, pred_Y)
    print "precision_score:"
    print metrics.precision_score(test_Y, pred_Y)
    print "confusion_matrix:"
    print metrics.confusion_matrix(test_Y, pred_Y)
    print "AUC:"
    print metrics.roc_auc_score(test_Y, pred_Y)



if __name__ == '__main__':
    func1()