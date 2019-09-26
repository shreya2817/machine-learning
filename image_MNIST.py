# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:58:54 2019

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_mldata
dataset = fetch_mldata('MNIST original')

dataset = pd.read_csv('ml datas/mnist_train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
                
X = dataset.data
y = dataset.target

some_digit = X[9993]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)
dtf.score(X_test, y_test)
dtf.score(X, y)


