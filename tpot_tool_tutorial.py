# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 23:15:52 2016

@author: Eng.Alla Khaled
"""

"""
1. Install dependencies
2. Create Classifier
3. Analyze Results
"""
from tpot import TPOTClassifier #a tool that automatically creates and optimizes machine learning pipelines using genetic programming. 
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

#load data
telescope = pd.read_csv('magic-telescope.csv') # a dataset that measures radiation in the atmosphere
#clean data
telescope_shuffle = telescope.iloc[np.random.permutation(len(telescope))] # iloc() is a pandas function to get positions in the index and will generate a sequence of random indicies
                                                                            # and will generate a sequence of random indices of the size of our data in telescope variavble using premutation()
tele = telescope_shuffle.reset_index(drop=True)

# store 2 classes
tele['Class'] = tele['Class'].map({'g':0, 'h':1}) # g for gamma & h for hydron which indicate 2 columns in the data set with integers 0 & 1
tele_class = tele['Class'].values

#split training, testing, and validation data
training, validation = training, testing = train_test_split(tele.index,
                 stratify = tele_class, train_size=0.75, test_size=0.25)

# let Genetic programming find best ML model and hyperparameters
tpot = TPOTClassifier(generations=5, verbosity=2) # it takes 5mins to generate 1 generation so it will take 25mins, 5mins is the default until it says Timeout
tpot.fit(tele.drop('Class', axis=1).loc[training].values,  # verbosity -> to show a progress bar of the ooptimization process
         tele.loc[training, 'Class'].values)
         
# score the accuracy
tpot.score(tele.drop('Class', axis=1).loc[validation].values,
         tele.loc[validation, 'Class'].values)       

# export the generated code         
tpot.export('pipeline.py')

"""TPOT will take a while because it's running k-fold cross-validation 
on every pipeline on the full training data set"""