
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
dataset = pd.read_csv('Desktop/ml datas/housing.csv')

pd.scatter_matrix(dataset)

X = dataset.iloc[:, [0,1,2,3,4,5,6,7,9]].values
y = dataset.iloc[:, 8].values
                
from sklearn.preprocessing import Imputer
imp = Imputer()
X[:, 0:8] = imp.fit_transform(X[:, 0:8])

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, 8] = lab.fit_transform(X[:, 8])

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [8])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_predict = lin_reg.predict(X_test)

lin_reg.score(X_train, y_train)
lin_reg.score(X_test, y_test)
lin_reg.score(X, y)

coef = lin_reg.coef_
bias = lin_reg.intercept_

lin_reg.predict(X_train [5])
lin_reg.predict(X_train[5].reshape(-1, 1))
lin_reg.predict(X_train [5 : 6])

X_train[5 : 7]

lin_reg.predict(X_train[5:7, :])