# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV



if __name__ == '__main__':
