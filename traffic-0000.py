
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:16:24 2019
@author: Amit Gupta
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DAY_RANGE = 15
TIME = '00:00:00'
# Importing the training set
metro_traffic = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')


# NO NA
sum(metro_traffic['traffic_volume'].isna())




# Converting datetime feature to datetime 
metro_traffic['date_time'] = pd.to_datetime(metro_traffic.date_time)

# Segrigating datetime to date & time feature 
metro_traffic['date'] = metro_traffic['date_time'].dt.date
metro_traffic['time'] = metro_traffic['date_time'].dt.time

# Converting time to string 
metro_traffic['time'] = metro_traffic['time'].astype(str)
# Converting time to string 
metro_traffic['date'] = metro_traffic['date'].astype(str)

# filtering only 9:00 oclock records
dataset=metro_traffic[metro_traffic['time'] == TIME]
dataset = dataset.drop_duplicates(subset=['date'], keep='first', inplace=False)

# divide the dataset into test and training dataset
dataset_train =  dataset[dataset['date'] <= '2018-09-01']
dataset_test =  dataset[dataset['date'] >= '2018-09-01']


training_set = dataset_train.iloc[:, 8:9].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(DAY_RANGE, len(dataset_train)):
    X_train.append(training_set_scaled[i-DAY_RANGE:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



training_set = dataset_train.iloc[:,8:9].values


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 150, return_sequences = True, input_shape = (None, 1)))
#regressor.add(Dropout(.1))
# Adding a second LSTM layer
#regressor.add(LSTM(units = 90, return_sequences = True))

# Adding a third LSTM layer
regressor.add(LSTM(units = 150, return_sequences = True))
#regressor.add(Dropout(.1))

regressor.add(LSTM(units = 150, return_sequences = True))
#regressor.add(Dropout(.1))


regressor.add(LSTM(units = 150, return_sequences = True))
#regressor.add(Dropout(.1))

#regressor.add(Dropout(.1))





# Adding a fourth LSTM layer
regressor.add(LSTM(units = 150))
#regressor.add(Dropout(.1))

# Adding the output layer
regressor.add(Dense(units = 1))
#regressor.add(Dropout(.1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 8)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
#dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 8:9].values

# Getting the predicted stock price of 2017
dataset_total = dataset.iloc[:, 8:9].values
inputs = dataset_total[len(dataset_total) - len(dataset_test) - DAY_RANGE:]

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

for i in range(DAY_RANGE, len(inputs)):
    X_test.append(inputs[i-DAY_RANGE:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real traffic')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted traffic')
plt.title('Traffic Prediction at ' + TIME + ' day range ' + str(DAY_RANGE) + ' Three regressor 90')
plt.xlabel('Date')
plt.ylabel('Traffic')
plt.legend()
plt.show()



from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(dataset_test['traffic_volume'], predicted_stock_price))
rms

#  157.551  15 days 190.871 90 days
