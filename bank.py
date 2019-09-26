# -*- coding: utf-8 -*-
"""
Created on Fri May 31 00:11:06 2019

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('ml datas/bank.csv')
X = dataset.iloc[:,0:16].values
y = dataset.iloc[:, -1].values
                 
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
 
X[:, 1] = lab.fit_transform(X[:, 1])
X[:, 2] = lab.fit_transform(X[:, 2])
X[:, 3] = lab.fit_transform(X[:, 3])
X[:, 4] = lab.fit_transform(X[:, 4])
X[:, 6] = lab.fit_transform(X[:, 6])
X[:, 7] = lab.fit_transform(X[:, 7])
X[:, 8] = lab.fit_transform(X[:, 8])
X[:, 10] = lab.fit_transform(X[:, 10])
X[:, 15] = lab.fit_transform(X[:, 15])               

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1, 2, 3, 4, 6, 7, 8, 10, 15])
X = one.fit_transform(X)
X = X.toarray()

y = lab.fit_transform(y)
lab.classes_

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_reg.score(X_train, y_train)
lin_reg.score(X_test, y_test)
lin_reg.score(X, y)

lin_reg.predict(X[20, :])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


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
knn.predict(x[20, :]) 

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)
dtf.score(X_test, y_test)
dtf.score(X, y)
