# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 21:46:13 2017

@author: Eng.Alla Khaled
"""

# Polynomial Regression

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
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train) """

# Fitting the Regression Model to the dataset

# predicting a new result with polynomila regression
y_pred = regressor.predict((6.5))

# Visualizing the regression model results

plt.scatter(x, y, color ='red')
plt.scatter(x, regressor.predict(x), color = 'blue', marker = '_')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Regression Model)')
plt.show()

# Visualizing the regression model results (For higher resolution scatter curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color ='red')
plt.scatter(x_grid, regressor.predict(x_grid), color = 'blue', marker = '_')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Regression Model)')
plt.show()
