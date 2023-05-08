# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

data9 = pd.read_excel('Dataset1/data_match9.xlsx', engine='openpyxl')
data10 = pd.read_excel('Dataset1/data_match10.xlsx', engine='openpyxl')

data9.dropna(inplace=True)
data10.dropna(inplace=True)


print('LSTM')
print('Train on Data10, validate on Data9')

X_train = data10[['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']]
y_train = data10['value']

X_test = data9[['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']]
y_test = data9['value']

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(1, X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print('epochs: ',200)
print('batch_size: ',128)
model.fit(X_train, y_train, epochs=200, batch_size=128, verbose=0)

predicted = model.predict(X_test)
print('R2 score:', r2_score(y_test, predicted))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predicted)))
print()