# -*- coding: utf-8 -*-
"""
Created on Fri May 05 15:48:34 2017

@author: Eng.Alaa Khaled
"""

# Data Preprocessing

# import libraries/ dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Data.csv")
# call the independent features
x = dataset.iloc[:, :-1].values # [:, :-1] means to take all rows, and all cols except for the last col
# call the dependent variable/ label
y = dataset.iloc[:,3].values

# take care of missing data,
# replace missing data by mean value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy="mean",axis=0)
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3]) #replaces missing by mean value of the column

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])

# Dummy Encoding
onehotencoder = OneHotEncoder(categorical_features=[0]) # encode the country col
x = onehotencoder.fit_transform(x).toarray()
# Encode the label/dependent variable yes:1 , no:0
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting data into training and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8,  random_state= 0)

# Feature scaling using [standaralization = (x-mean(x)/ SD(x)) or Normalization = (x-min(x))/ max(x)-min(x)]
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
# we don't need to aplly feature scaiiling for the dependent variable for a binary classifier

