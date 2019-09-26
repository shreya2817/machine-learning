# -*-coding: utf-8 -*-
"""
Created on Tue May 14 11:59:12 2019

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Desktop/ml datas/Data_Pre.csv')
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,-1].values
                
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN',strategy = 'mean', axis= 0)
imp.fit(X[:,0:2])
X[:,0:2] = imp.transform(X[:, 0:2])

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, 2] = lab.fit_transform(X[:, 2])
lab.classes_

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [2])
X = one.fit_transform(X)
X = X.toarray()
y = lab.fit_transform(y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

