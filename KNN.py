# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import cross_validate as cv
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR as svr
from sklearn.neighbors import KNeighborsRegressor

## Dataset 1
data9 = pd.read_excel('Dataset1/data_match9.xlsx', engine='openpyxl')
data10 = pd.read_excel('Dataset1/data_match10.xlsx', engine='openpyxl')

data9.dropna(inplace=True)
data10.dropna(inplace=True)

print('KNN, K=128')
print('Train on Data10, validate on Data9')

X_train = data10[['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']]
y_train = data10['value']
X_val = data9[['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']]
y_val = data9['value']

knn = KNeighborsRegressor(n_neighbors=128)
knn.fit(X_train, y_train)

data9['predicted'] = knn.predict(X_val)
print('R2 score:', knn.score(X_val, y_val))
print('RMSE:', np.sqrt(mse(data9['predicted'], y_val)))
knn = None
print()

