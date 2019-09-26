# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:54:02 2019

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel("Desktop/ml datas/blood.xlsx")
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, -1].values
X = X.reshape(-1,1)                

plt.scatter(X, y)
plt.show()
             
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
          
plt.scatter(X, y)
plt.plot(X, lin_reg.predict(X), c ="g")
plt.show()   