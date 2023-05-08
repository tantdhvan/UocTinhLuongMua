# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model

## Dataset 1
data9 = pd.read_excel('Dataset1/data_match9.xlsx', engine='openpyxl')
data10 = pd.read_excel('Dataset1/data_match10.xlsx', engine='openpyxl')

data9.dropna(inplace=True)
data10.dropna(inplace=True)

print('BayesianRidge')
print('Train on Data10, validate on Data9')

X_train = data10[['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']]
y_train = data10['value']
X_test = data9[['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']]
y_test = data9['value']

reg = linear_model.BayesianRidge()
reg.fit(X_train, y_train)

pre = reg.predict(X_test)
print('R2 score:', reg.score(X_test, y_test))
print('RMSE:', np.sqrt(mse(pre, y_test)))

