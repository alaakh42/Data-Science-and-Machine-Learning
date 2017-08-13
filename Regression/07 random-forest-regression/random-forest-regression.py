# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:36:31 2017

@author: Eng.Alla Khaled
"""

# random Forest Regression 

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

# Fitting the RF Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, criterion = "mse", random_state = 0)
regressor.fit(x, y )

# predicting a new result with RF regression model
y_pred = regressor.predict((6.5))


"""
We need to make the Random forest Regression
Model in a higher resolution because it is 
a nonlinear noncontinous regession model
"""

# Visualizing the RF regression model results (For higher resolution scatter curve)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color ='red')
plt.scatter(x_grid, regressor.predict(x_grid), color = 'blue', marker = '_')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.show()
