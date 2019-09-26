import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open("dataset.pkl",'rb') as f:
    Xd = pickle.load(f, encoding = 'latin1')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1. 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]): lbl.append((mod, snr))\
X = np.vstack(X)

np.random.seed(2016)
n_exaamples = X.shape[0]
n_train = int(n_examples * 0.5)

train_idx = np..random.choice(range(0, n_examples), size = n_train, replace= False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
X_train =X[train_idx]
X_test = X[test_idx]

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1

Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))