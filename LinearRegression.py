import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import cross_validate as cv
from sklearn.metrics import mean_squared_error as mse

## Dataset 1
data9 = pd.read_excel('Dataset1/data_match9.xlsx', engine='openpyxl')
data10 = pd.read_excel('Dataset1/data_match10.xlsx', engine='openpyxl')

data9.dropna(inplace=True)
data10.dropna(inplace=True)

print('LinearRegression ')
print('Train on Data10, validate on Data9')

X_train = data10[['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']]
y_train = data10['value']
X_val = data9[['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']]
y_val = data9['value']

lr_model = lr().fit(X_train, y_train)
pre = lr_model.predict(X_val)

print('R2 score:', lr_model.score(X_val, y_val))
print('RMSE:', np.sqrt(mse(pre, y_val)))
lr_model = None
print()