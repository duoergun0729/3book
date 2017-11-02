# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import lief

def func1():
    h = FeatureHasher(n_features=3)
    D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
    f = h.transform(D)
    print(f.toarray())

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    print("shape:")
    print(shape)
    print("strides:")
    strides = a.strides + (a.strides[-1],)
    print(strides)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def func2():
    x = np.arange(16)
    print(x)

    blocks = rolling_window(x, 2)[::1, :]
    #print(blocks)
    for block in blocks:
        print(block)
        #c=block>>2
        #c = np.bincount(block >> 2, minlength=16)
        #print(c)
        #wh = np.where(c)[0]
        #print(wh)
        #print(".......")

def func3():
    x, y = datasets.make_classification(n_samples=1000, n_features=100,n_redundant=0, random_state = 1)

    knn = KNeighborsClassifier(n_neighbors=5)
    score1 = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
    print(np.mean(score1))


    gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth = 1, random_state = 1)
    score2 = cross_val_score(gbdt, x, y, cv=5, scoring='accuracy')
    print(np.mean(score2))

    xgboost = xgb.XGBClassifier()
    score3 = cross_val_score(xgboost, x, y, cv=5, scoring='accuracy')
    print(np.mean(score3))


def func4():
    x, y = datasets.make_classification(n_samples=1000, n_features=100, n_redundant=0, random_state=1)
    parameters ={ 'n_estimators':range(50,200,25) }
    gsearch = GridSearchCV(
        estimator=GradientBoostingClassifier(),
        param_grid=parameters, scoring='accuracy', iid=False, cv=5)
    gsearch.fit(x, y)
    #print("gsearch.cv_results_")
    #print(gsearch.cv_results_)
    print("gsearch.best_params_")
    print(gsearch.best_params_)
    print("gsearch.best_score_")
    print(gsearch.best_score_)

    print(gsearch.grid_scores_)

def func5():
    x, y = datasets.make_classification(n_samples=1000, n_features=100, n_redundant=0, random_state=1)
    parameters ={ 'n_estimators':range(50,200,25), 'max_depth':range(2,10,2)}
    gsearch = GridSearchCV(
        estimator=GradientBoostingClassifier(),
        param_grid=parameters, scoring='accuracy', iid=False, cv=5)
    gsearch.fit(x, y)

    print("gsearch.best_params_")
    print(gsearch.best_params_)
    print("gsearch.best_score_")
    print(gsearch.best_score_)

def func6():
    pefile="a1303f026b713fbe7fe165cc8609847f5ec46bb2dfdbe86cff4b12deae728ca3"
    binary = lief.parse(pefile)
    #print(binary.dos_header)
    print(binary.header)
    #print(binary.optional_header)

    #for func in binary.imported_functions:
    #    print(func)

def func7():
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)

    distances, indices = nbrs.kneighbors(X)
    print(indices)
    print(distances)

if __name__ == '__main__':
    func7()