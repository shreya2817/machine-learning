# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:52:00 2019

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Desktop/ml datas/DemographicData.csv')
X = dataset.iloc[:,2:4].values
z = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()                
z = lab.fit_transform(z)
lab.classes_

plt.scatter([1,2,3],[3,2,1])
plt.scatter(X[z == 0,0],X[z == 0,1], c='r', label = "High Income")
plt.scatter(X[z == 1,0],X[z == 1,1], c='b', label = "low Income")
plt.scatter(X[z == 2,0],X[z == 2,1], c='g', label = "lower middle Income")
plt.scatter(X[z == 3,0],X[z == 3,1], c='y', label = "upper middle Income")
plt.legend()
plt.xlabel("birth rate")
plt.ylabel("internet users")
plt.title("a case study of relationship between birth rate and internet users" )
plt.show()