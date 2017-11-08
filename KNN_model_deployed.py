# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:47:23 2017

@author: Hopeless
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import pandas as pd
import numpy as np
import itertools
import math
import ast
import os

# Reading the dataset
books = pd.read_csv(os.getcwd() + "/booksummaries_hot_encoded.txt", sep=';')

target = books.iloc[:,1:2]
data = books.iloc[:,4:]

# Splitting the dataset into training set and test set
X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(data.as_matrix(), target.as_matrix(), test_size = 0.20, random_state = 42)

# print (X_Train.shape)
# print (X_Test.shape)
# print (Y_Train.shape)
# print (Y_Test.shape)

# KNN Model to recommend n number of books for a test case
KNN = KNeighborsClassifier(n_neighbors = 7)
KNN.fit(X_Train, Y_Train)

distances, neighbor = KNN.kneighbors(X_Test)

# Sample test book
# print(Y_Test[0])

# n number of similar books for given sample 
# print(neighbor[0])

# Book names of similar books for given sample
# print(Y_Train[neighbor[0]])

# Saving the trained module
import pickle
save_classifier = open(os.getcwd() + "/knn_classifer.pickle","wb")
pickle.dump(KNN, save_classifier)
save_classifier.close()