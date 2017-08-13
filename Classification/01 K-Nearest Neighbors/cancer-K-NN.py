# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:33:19 2017

@author: Eng.Alla Khaled
"""

# K-Nearest Neighbors (K-NN)

# import dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Breast_Cancer.csv")
dataset.replace('?',-99999, inplace=True) # replace missing data
dataset.drop(['id'], 1, inplace=True) # drop id col
x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:,9].values
"""2-> benign and 4-> maliginant"""


# splitting data into training and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8,  random_state= 0)

## Feature scaling using [standaralization = (x-mean(x)/ SD(x)) or Normalization = (x-min(x))/ max(x)-min(x)]
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#x_train = sc_x.fit_transform(x_train)
#x_test = sc_x.transform(x_test)

# Fitting the classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2 )
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Compute the Confusion matrix (Classifier Evaluation)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
""" TN = 108, FP = 4, FN = 4, TP = 59 """

# Calculate the K-NN Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)*100   # 95.714%
"""I got an acc = 98.57% without doing any feature scaling"""

# make a prediction on an outsider data instance
ex = np.array([[1,2,3,4,2,2,2,3,4],[5,6,72,4,3,5,6,22,1]]) # this is a random instance i crated
ex = ex.reshape(len(ex), -1)
prediction = classifier.predict(ex) # 2 as benign,  4 as maliginant


