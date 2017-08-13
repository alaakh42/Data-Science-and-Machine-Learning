# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:54:42 2017

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
x_test = sc_x.transform(x_test) """

# Fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg2 = LinearRegression() # Create the Polynomial model
lin_reg2.fit(x_poly, y)

# Visualizing the Linear regresison results
plt.scatter(x, y, color ='red') # plot the true y 'real salaries'/ it is parabolic
plt.scatter(x, lin_reg.predict(x), color = 'blue', marker='*' ) # plot the predicted y 'predicted salaries'/
                                                                # it is linear as it is the prediction of the linear regression model
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Linear Regression)')
plt.show()

# Visualizing the Polynomial regression results
x_grid = np.arange(min(x), max(x), 0.1) # include all levels from 1 to 10 with an increament of 0.1
x_grid = x_grid.reshape((len(x_grid), 1)) # form all those values in a matrix 
plt.scatter(x, y, color ='red')
#plt.scatter(x, lin_reg2.predict(poly_reg.fit_transform(x)), color = 'blue', marker = '^')
plt.scatter(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color = 'blue', marker = '_')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.show()

# predicting a new result with Linear regression
lin_reg.predict(6.5) # return the salary for a 6.5 level according the LR model

# predicting a new result with polynomila regression
lin_reg2.predict(poly_reg.fit_transform(6.5)) # return the salary for a 6.5 level according the Polynomial regression model

