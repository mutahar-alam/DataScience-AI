# -*- coding: utf-8 -*-
"""
Created on Fri May 18 01:00:17 2018

@author: mmalam
"""

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('breast-cancer-wisconsin-data.csv',header=0)
dataset = pd.get_dummies(dataset, columns=['diagnosis'])

dataset = dataset.drop(['id','Unnamed: 32', 'diagnosis_B'], axis=1)

X = dataset.iloc[:, 0:30].values
y = dataset.iloc[:, 30].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


##Random Forest implementation
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)
predicted_y_RF = regressor.predict(X_test)
cm2 = confusion_matrix(y_test, predicted_y_RF)


model = Sequential()
model.add(Dense(units=15,kernel_initializer='uniform',activation='relu',input_dim=30))
model.add(Dense(units=15,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=75, batch_size=10)

predicted_y = model.predict(X_test)
predicted_y[:] = (predicted_y[:] > .2)

cm = confusion_matrix(y_test, predicted_y)
