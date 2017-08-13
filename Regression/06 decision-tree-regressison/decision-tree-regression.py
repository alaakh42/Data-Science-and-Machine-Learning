# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 23:25:50 2017

@author: Eng.Alla Khaled
"""
# Decission Tree Regression

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:,2].values 

# Fitting the Decision Tree Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion = 'mse', random_state = 0)
regressor.fit(x, y)

# predicting a new result with Decision Tree Regression 
y_pred = regressor.predict((6.5)) # 150,000

"""
We need to make the Decision tree Regression
Model visualization in a higher resolution because
it is a nonlinear noncontinous regession model
"""

# Visualizing the regression model results (For higher resolution scatter curve)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color ='red')
plt.scatter(x_grid, regressor.predict(x_grid), color = 'blue', marker="_")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Decision Tree Regression Model)')
plt.show()
