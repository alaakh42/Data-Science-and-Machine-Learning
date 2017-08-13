
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:34:32 2017

@author: Eng.Alla Khaled
"""

# Random Forest Classification

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# Claculate the processing time
start_time = time.time()

# Importing the dataset (credit-card)
dataset = pd.read_csv('dataset.csv')
#X = dataset.iloc[:,:-1].values
X = dataset.iloc[:,[0,1,2,5,6,7,10,13,14,15]].values
y = dataset.iloc[:, 16].values

# Importing the dataset (Telecom)
#dataset = pd.read_csv('newlabeleddataset.csv')
##X = dataset.iloc[:,:-1].values
#X = dataset.iloc[:, [0,2,3,5,6,9]]
#y = dataset.iloc[:, 10].values



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0, n_jobs = 4)
classifier.fit(X_train, y_train)
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
For Credit Card data:::::
(Cnfusion Matrix)
array([[ 19,   1],
       [  0, 327]])
TN = 19
FP = 1
FN = 0
TP = 327
There is only 2+0 =2 incorrect predictions
NOTE: For Credit Card data
1 fraud ----> negative class
2 non-fraud -----> positive class
################################################
For Telecom Data:::::
(Cnfusion Matrix)
array([[350,   5],
       [ 50,  12]])
TN = 350
FP = 5
FN = 50
TP = 12
There is only 5+50 =55 incorrect predictions
NOTE: For Telecom data
1 fraud ----> positive class
0 non-fraud -----> negative class
"""

# Calculate the precision, recall and F1_Score scores
from sklearn import metrics
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)
"""
For Telecom Data
precision: 0.70588235294117652
recall: 0.19354838709677419
f1_score: 0.30379746835443039

For Credit Card
precision: 1.0
recall: 0.94999999999999996
f1_score: 0.97435897435897434
"""
# Calculate Random Forest Classifier Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)*100
""" For Telecom data, Accuracy= 86.810551558753005% """
""" For credit card data, Accuracy= % """

print("--- %s seconds ---" % (time.time() - start_time))


# Calculate Features Importance of the Training Features and Plot them
importances = classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
#plt.barh(range(X.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center") # plot a hirozontal bars
plt.xticks(range(X.shape[1]), indices) # use the features indices in the feature importance graph
feature_names = dataset.columns # use the features names in the feature importance graph
#plt.xticks(range(X.shape[1]), feature_names)
plt.xlim([-1, X.shape[1]])
plt.show()


# Print the OOB error for the RF Classifier for every estimator
"""
n_estimators = 100
forest = RandomForestClassifier(warm_start=True, oob_score=True)

for i in range(1, n_estimators + 1):
    forest.set_params(n_estimators=i)
    forest.fit(X, y)
    print i, forest.oob_score_
"""  
#  the OOB error can be measured at the addition of each new tree during training.
# The resulting plot allows a practitioner to approximate a suitable value of n_estimators
# at which the error stabilizes.
#from collections import OrderedDict
#RANDOM_STATE = 123  
## NOTE: Setting the `warm_start` construction parameter to `True` disables
## support for parallelized ensembles but is necessary for tracking the OOB
## error trajectory during training.
#ensemble_clfs = [
#    ("RandomForestClassifier, max_features='sqrt'",
#        RandomForestClassifier(warm_start=True, oob_score=True,
#                               max_features="sqrt",
#                               random_state=RANDOM_STATE)),
#    ("RandomForestClassifier, max_features='log2'",
#        RandomForestClassifier(warm_start=True, max_features='log2',
#                               oob_score=True,
#                               random_state=RANDOM_STATE)),
#    ("RandomForestClassifier, max_features=None",
#        RandomForestClassifier(warm_start=True, max_features=None,
#                               oob_score=True,
#                               random_state=RANDOM_STATE))
#]
#
## Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
#error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
#
## Range of `n_estimators` values to explore.
#min_estimators = 15
#max_estimators = 175
#
#for label, clf in ensemble_clfs:
#    for i in range(min_estimators, max_estimators + 1):
#        clf.set_params(n_estimators=i)
#        clf.fit(X, y)
#
#        # Record the OOB error for each `n_estimators=i` setting.
#        oob_error = 1 - clf.oob_score_
#        error_rate[label].append((i, oob_error))
#
## Generate the "OOB error rate" vs. "n_estimators" plot.
#for label, clf_err in error_rate.items():
#    xs, ys = zip(*clf_err)
#    plt.plot(xs, ys, label=label)
#
#plt.xlim(min_estimators, max_estimators)
#plt.xlabel("n_estimators")
#plt.ylabel("OOB error rate")
#plt.legend(loc="upper right")
#plt.show()


"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 3].min() - 1, stop = X_set[:, 3].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 4].min() - 1, stop = X_set[:, 4].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 5].min() - 1, stop = X_set[:, 5].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 6].min() - 1, stop = X_set[:, 6].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 7].min() - 1, stop = X_set[:, 7].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 8].min() - 1, stop = X_set[:, 8].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 9].min() - 1, stop = X_set[:, 9].max() + 1, step = 0.01))
#                     np.arange(start = X_set[:, 10].min() - 1, stop = X_set[:, 10].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 11].min() - 1, stop = X_set[:, 11].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 12].min() - 1, stop = X_set[:, 12].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 13].min() - 1, stop = X_set[:, 13].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 14].min() - 1, stop = X_set[:, 14].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 15].min() - 1, stop = X_set[:, 15].max() + 1, step = 0.01))
              
plt.contourf(X1, X2, , X3, X4, X5, X6, X7, X8, X9, X10,classifier.predict(np.array([X1.ravel(), X2.ravel(), , X3.ravel(), , X4.ravel(),, X5.ravel(),, X6.ravel(),, X7.ravel(),, X8.ravel(),, X9.ravel(),, X10.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

"""