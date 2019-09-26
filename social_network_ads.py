# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 20:03:21 2019

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train, y_train)
svc.score(X_train, y_train)
svc.score(X_test, y_test)
svc.score(X, y)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svc.score(X_train, y_train)
svc.score(X_test, y_test)
svc.score(X, y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)

y_pred = log_reg.predict(X_test) 

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_train, y_train)
knn.score(X_test, y_test)
knn.score(X, y)
knn.predict(X[20, :]) 

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(random_state=0)
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)
dtf.score(X_test, y_test)
dtf.score(X, y)

