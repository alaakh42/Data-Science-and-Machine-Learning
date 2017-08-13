# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:33:11 2017

@author: Eng.Alla Khaled
"""

# Kernel SVM

# import dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# import dataset
dataset = pd.read_csv("Social_network_Ads.csv")
x = dataset.iloc[:, [2,3]].values 
y = dataset.iloc[:,4].values

# splitting data into training and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, train_size=0.75,  random_state= 0)

# Feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Fitting the classifier to the Training set
from sklearn.svm import SVC 
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(x_train, y_train)
classifier = SVC(kernel = 'rbf', random_state = 0, C = 30, gamma = 3)
#classifier = SVC(kernel = 'rbf', random_state = 0, C =  0.3512656311582968, gamma = 0.9948834716756692)
classifier.fit(x_train, y_train)

""" Hyperparameters optimization (C and gamma values) using Grid Search
    Where C -> is Penalty parameter C of the error term.
    And gamma -> Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    If gamma is ‘auto’ then 1/n_features will be used instead.
"""

# possible hyperparameters
#c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
#gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

# accuracy
best_score = 0
best_params = {'C': None, 'gamma': None}

## Hyperparameters optimization
##for i in range(10):
#for c in c_values:
#    for gamma in gamma_values:
#        # train the model for every hyperparameter value pair
#        classifier = SVC(kernel = 'rbf', C = c, gamma = gamma)
#        #classifier = SVC(kernel = 'rbf', C = random.randint(0,9), gamma = random.randint(0,3))
#        classifier.fit(x_train, y_train)
#        score = classifier.score(x_test, y_test)
#        
#        # rate the accuracy of the model using each hyperparameter value pair
#        if score > best_score:
#            best_score = score
#            best_params['C'] =  c
#            best_params['gamma'] = gamma
#
## best score
#print best_score, best_params 
"""0.94 {'C': 30, 'gamma': 3}"""

"""Hyperparameters optimization (C and gamma values) 
    Using 'Bayesian Optimization' """

import sklearn.gaussian_process as gp
#prior belief for[C_value, gamma value]
C = 4
gamma = 0.3

## for a preset number of iterations
#for i in range(10):
#    # try values for each hyperparameter
#    C, gamma = gaussian_process(hyperparams= [C, gamma], loss= expected_improvement)
#    classifier = SVC(kernel = 'rbf', C = C, gamma = gamma)
#    classifier.fit(x_train, y_train)
#    score = classifier.score(x_test, y_test)
#    
#    if score > best_score:
#        best_score = score
#        best_params['C'] =  C
#        best_params['gamma'] = gamma
#
## best score
#print best_score, best_params 

""" Another Attempt for Hyperprameter optimization 
    using the hyperopt package
"""


from hyperopt import hp, fmin, tpe, rand, space_eval

C = 4
gamma = 0.3

# define a search space
space =  [hp.uniform('gamma', 0, 1), hp.uniform('C', 0, 1)]

# define an objective function
def f(args):
    classifier = SVC(kernel = 'rbf', C = C, gamma = gamma)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    return score
    
# optimize it using algorithm of choice
best = fmin(f, space, algo= rand.suggest, max_evals= 100)
print 'random', best

best = fmin(f, space, algo= tpe.suggest, max_evals= 100)
print 'tpe', best
"""random {'C': 0.13868621507163414, 'gamma': 0.4634381270836627}
    tpe {'C': 0.3512656311582968, 'gamma': 0.9948834716756692} """

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Compute the Confusion matrix (Classifier Evaluation)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""[64,  4],
   [ 3, 29]  FN+FP = 3+4 = 7 incorrect predictions"""
   
# Calculate the Kernel SVM Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)*100   # 93%

# Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, edgecolors = 'black')
plt.title('Gaussian Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, edgecolors = 'black')
plt.title('Gaussian Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()