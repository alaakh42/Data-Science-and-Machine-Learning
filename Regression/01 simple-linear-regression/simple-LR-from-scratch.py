# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 14:16:36 2017

@author: Eng.Alla Khaled
"""
from math import sqrt
"""
y = b0 + b1*x   #simple linear regression equation
b1 = covariance(x,y)/ variance(x)
b0 = mean(y) - b1*mean(x)
"""
 
# Calculate the mean value of a list of numbers
def mean(values):
    return sum(values)/ float(len(values))
    
# Calculate the variance of a list of numbers    
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

# Calculate the covariance between x and y       
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x)*(y[i] - mean_y)
    return covar
    
# calculate the coeffecients
def coefficients(dataset):
    x = [col[0] for col in dataset]
    y = [col[1] for col in dataset]
    mean_x = mean(x)
    mean_y = mean(y)
    b1 = covariance(x, mean_x, y, mean_y)/ variance(x, mean_x)
    b0 = mean_y - b1 * mean_x
    return [b0, b1]
    
# Simple LR algorithm
def SLR(train, test):
    predictions = []
    b0, b1 = coefficients(train)
    for i in test:
        yhat = b0 + b1 * i[0]
        predictions.append(yhat)
    return predictions
 
# calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error =  predicted[i] - actual[i]
        sum_error += (prediction_error**2)
    mean_error = sum_error/ float(len(actual))
    return sqrt(mean_error)
    
# Evaluate regression algorithm on training dataset
def evaluate_algorithm(dataset, algorithm):
    test_set = []
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(dataset, test_set)
    print (predicted)
    actual = [row[-1] for row in dataset]
    rmse = rmse_metric(actual, predicted)
    return rmse

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0    
# Test SLR
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
rmse = evaluate_algorithm(dataset, SLR)
print('RMSE: %.3f' % (rmse))     