# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:41:15 2019

@author: hp
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)

y_pred = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import precision_score, recall_score
precision_score(y_test, y_pred, average='micro')
recall_score(y_test, y_pred, average='micro')

precision_score(y_test, y_pred, average='macro')
recall_score(y_test, y_pred, average='macro')