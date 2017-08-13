# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:10:59 2017

@author: Eng.Alla Khaled
"""

# Support Vector Regression (SVR)

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:,2].values 

# splitting data into training and test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8,  random_state= 0)"""

# Feature scaling using [standaralization = (x-mean(x)/ SD(x)) or Normalization = (x-min(x))/ max(x)-min(x)]
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y) 

# Fitting the Support Vector Regression Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

# Visualizing the Support Vector regression model results
plt.scatter(x, y, color ='red')
plt.scatter(x, regressor.predict(x), color = 'blue', marker = '_')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (SVR Model)')
plt.show()

# predicting a new result with Support Vector regression

# we need to transform the '6.5' into an np matrix/ array
y_pred = regressor.predict(sc_x.transform(np.array([6.5]))) 
# then we need to inverse the feature scalling we did on the prediction to return to the original scale of the original y values
y_pred = sc_y.inverse_transform(y_pred) # """y_pred = array([ 170370.0204065])"""
 
# Visualizing the regression model results (For higher resolution scatter curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color ='red')
plt.scatter(x_grid, regressor.predict(x_grid), color = 'blue', marker = '_')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (SVR Model)')
plt.show()
