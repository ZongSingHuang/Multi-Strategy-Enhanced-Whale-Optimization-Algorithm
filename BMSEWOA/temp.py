# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import MSEWOA2
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
np.random.seed(42)
# -----------------------------------------------------------------------------


# 參數
np.random.seed(42)
Zoo = pd.read_csv('Zoo.csv', header=None).values
X_train, X_test, y_train, y_test = train_test_split(Zoo[:, :-1], Zoo[:, -1], stratify=Zoo[:, -1], test_size=0.5)
# -----------------------------------------------------------------------------


def Zoo_test(x):
    x = x.astype(int)
    
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    loss = np.zeros(x.shape[0])
    loss_error = np.zeros(x.shape[0])
    loss_feature = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        if np.sum(x[i, :])>0:
            knn = KNeighborsClassifier(n_neighbors=5).fit(X_train[:, x[i, :]], y_train)
            score = accuracy_score(knn.predict(X_test[:, x[i, :]]), y_test)
            loss[i] = 0.99*(1-score) + 0.01*(np.sum(x[i, :])/X_train.shape[1])
            loss_error[i] = 1-score
            loss_feature[i] = np.sum(x[i, :])/X_train.shape[1]
        else:
            loss[i] = np.inf
            loss_error[i] = np.inf
            loss_feature[i] = np.inf
            print(666)
    return loss, loss_error, loss_feature
# -----------------------------------------------------------------------------

cost = time.time()
feature_size = X_train.shape[1]
x_max = 10*np.ones(feature_size)
x_min = -10*np.ones(feature_size)
optimizer = MSEWOA2.MSEWOA(fit_func=Zoo_test, num_dim=feature_size, num_particle=5, max_iter=70, 
                           x_max=x_max, x_min=x_min, method='modif_s')
optimizer.opt()
optimizer.plot_curve()
optimizer.plot_curve_plus()
cost = time.time() - cost
print(cost)
# -----------------------------------------------------------------------------


feature = Zoo[:, :-1]
label = Zoo[:, -1]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[:, optimizer.final.astype(int)], y_train)
print(accuracy_score(knn.predict(X_test[:, optimizer.final.astype(int)]), y_test))