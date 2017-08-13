# -*- coding: utf-8 -*-
"""
Created on Fri May 05 22:08:24 2017

@author: Eng.Alla Khaled
"""
# import libraries/ dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Salary_Data.csv")
# call the independent features
x = dataset.iloc[:,:-1].values # or iloc[:,0] # Here x is a matrix of 1 col
# call the dependent variable/ label
y = dataset.iloc[:,1].values # y is a vector


# splitting data into training and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state= 0)

# the simple linear regression library will take care of feature scaling for us
"""
# Feature scaling using [standaralization = (x-mean(x)/ SD(x)) or Normalization = (x-min(x))/ max(x)-min(x)]
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
# we don't need to aplly feature scaiiling for the dependent variable(y) for a binary classifier
"""
# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test) # contains the predicted values

plt.scatter(x_train, y_train, color='red') # real training values
plt.plot(x_train, regressor.predict(x_train) , color='blue') # predicted salieries
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color='red') # real testing values
plt.plot(x_train, regressor.predict(x_train) , color='blue') # predicted salieries
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
