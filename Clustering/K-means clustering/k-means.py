# -*- coding: utf-8 -*-
"""
Created on Sun Aug 06 20:04:42 2017

@author: Eng.Alla Khaled
"""
# K-Means Clustering

# import dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3,4]].values 

# Using the elbow method to find the optimal number of clusters K
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    Kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init = 10, random_state = 0)
    Kmeans.fit(x)
    wcss.append(Kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-means to the mall dataset
Kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter=300, n_init = 10, random_state = 0)
y_kmeans = Kmeans.fit_predict(x)

# Visualizing the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful', edgecolors='black')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard', edgecolors='black')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target', edgecolors='black')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless', edgecolors='black')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible', edgecolors='black')
plt.scatter(Kmeans.cluster_centers_[:,0], Kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids', edgecolors='black')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc = 'center right')
plt.show()   