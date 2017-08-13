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
dataset = pd.read_csv("50_Startups.csv")
# call the independent features
x = dataset.iloc[:, :-1].values # [:, :-1] means to take all rows, and all cols except for the last col
# call the dependent variable/ label
y = dataset.iloc[:,4].values

"""
# take care of missing data,
# replace missing data by mean value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy="mean",axis=0)
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3]) #replaces missing by mean value of the column
"""

# Encode categorical data into numerical, we will do this for the State col
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])
# Dummy Encoding
onehotencoder = OneHotEncoder(categorical_features=[3]) # encode the country col
x = onehotencoder.fit_transform(x).toarray()

# Avoiding the Dummy variable Trap, to do this
# all what you need is to not use one of the dummy variable columns
x = x[:, 1:] # we will take all coumns except for the first one with index 0


# splitting data into training and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8,  random_state= 0)

"""
# Feature scaling using [standaralization = (x-mean(x)/ SD(x)) or Normalization = (x-min(x))/ max(x)-min(x)]
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
# we don't need to aplly feature scaiiling for the dependent variable for a binary classifier
"""
# Fitting Multiple Linear Regressin on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) # fit the multiple LR to our trainig dataset

# predicting the Test set results
y_pred = regressor.predict(x_test) 

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1) # add a column of 50 1's to the x matrix
                                                                        # but we need to append it at the start of x matrix, 
                                                                        #so we 'll inverse arr and values values so we add the x matrix values to the column of 1's and not the opposite
x_optimal = x[:,[0,1,2,3,4,5]] # contains the optimal team of optimal independent variables
# fit the LR model to our future x_optimal matrix, we will create a new regressor using statsmdodel library specifically the A simple ordinary least squares model class
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()

x_optimal = x[:,[0,1,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()

x_optimal = x[:,[0,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()

x_optimal = x[:,[0,3,5]] 
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()

x_optimal = x[:,[0,3]] 
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()
