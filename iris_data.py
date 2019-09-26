# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:59:48 2019

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

#sepal length and sepal width
plt.scatter(X[y == 0, 0], X[y == 0, 1], c = "r", label = "setosa")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c = "g", label = "versicolor")
plt.scatter(X[y == 2, 0], X[y == 2, 1], c = "b", label = "verginica")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title("iris data")
plt.legend()
plt.show()

#sepal length and petal length
plt.scatter(X[y == 0, 0], X[y == 0, 2], c = "r", label = "setosa")
plt.scatter(X[y == 1, 0], X[y == 1, 2], c = "g", label = "versicolor")
plt.scatter(X[y == 2, 0], X[y == 2, 2], c = "b", label = "verginica")
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.title("iris data")
plt.legend()
plt.show()

#sepal length and petal width
plt.scatter(X[y == 0, 0], X[y == 0, 3], c = "r", label = "setosa")
plt.scatter(X[y == 1, 0], X[y == 1, 3], c = "g", label = "versicolor")
plt.scatter(X[y == 2, 0], X[y == 2, 3], c = "b", label = "verginica")
plt.xlabel("sepal length")
plt.ylabel("petal width")
plt.title("iris data")
plt.legend()
plt.show()

#sepal length and sepal length
plt.scatter(X[y == 0, 1], X[y == 0, 0], c = "r", label = "setosa")
plt.scatter(X[y == 1, 1], X[y == 1, 0], c = "g", label = "versicolor")
plt.scatter(X[y == 2, 1], X[y == 2, 0], c = "b", label = "verginica")
plt.xlabel("sepal width")
plt.ylabel("sepal length")
plt.title("iris data")
plt.legend()
plt.show()

#sepal width and petal length
plt.scatter(X[y == 0, 1], X[y == 0, 2], c = "r", label = "setosa")
plt.scatter(X[y == 1, 1], X[y == 1, 2], c = "g", label = "versicolor")
plt.scatter(X[y == 2, 1], X[y == 2, 2], c = "b", label = "verginica")
plt.xlabel("sepal width")
plt.ylabel("petal length")
plt.title("iris data")
plt.legend()
plt.show()

#sepal width and petal width
plt.scatter(X[y == 0, 1], X[y == 0, 3], c = "r", label = "setosa")
plt.scatter(X[y == 1, 1], X[y == 1, 3], c = "g", label = "versicolor")
plt.scatter(X[y == 2, 1], X[y == 2, 3], c = "b", label = "verginica")
plt.xlabel("sepal width")
plt.ylabel("petal width")
plt.title("iris data")
plt.legend()
plt.show()
