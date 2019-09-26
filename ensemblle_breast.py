# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:31:41 2019

@author: hp
"""

import numpy as np
import matplotlib.pyplot as pd
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset =load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

#K fol cross validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(log_reg, X_train, y_train, cv = 3)

log_reg.fit(X_train, y_train)
log_reg.score(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

#parameter grid from grid
param = [{'criterion' : ['gini', 'entropy']}, {'max_depth' : [3, 4, 5, 6, 7, 8, 9]}]

# grid search to estimate the best parameter
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(dtf, param,verbose=3) #verbose provides the info
grid.fit(X_train, y_train)

grid.best_estimator_
grid.best_index_
grid.best_params_
grid.best_score_
results = grid.cv_results_

#for score, param in zip(grid['mean_test_score'], grid['params']):
 #   print 
    
 
 from sklearn.svm import SVC
 svm = SVC()
 
 from sklearn.naive_bayes import GaussianNB
 n_b = GaussianNB()
 
log_reg.fit(X_train, y_train)
dtf.fit(X_train, y_train)
svm.fit(X_train, y_train)
n_b.fit(X_train, y_train)

log_reg.score(X_train, y_train)
dtf.score(X_train, y_train)
svm.score(X_train, y_train)
n_b.score(X_train, y_train)

#ensemble
from  sklearn.ensemble import VotingClassifier
vot = VotingClassifier ([('Log', log_reg), ('DT', dtf), ('SVN', svm), ('NB', n_b)])

vot.fit(X_train, y_train)
vot.score(X_train, y_train)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(log_reg, n_estimators = 5)
bag.fit(X_train, y_train)
bag.score(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 5)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
