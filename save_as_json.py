# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 04:36:27 2017

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
import pickle
import json
import re

# Load Classifier
classifier_new = open(os.getcwd() + "/knn_classifer.pickle","rb")
classifier = pickle.load(classifier_new)
classifier_new.close()

# Import dataset
books = pd.read_csv(os.getcwd() + "/booksummaries_hot_encoded.txt", sep=';')

# Split the dataset into its respective data and target values
target = books.iloc[:,1:2]
data = books.iloc[:,4:]

# Split data into train and test sets
X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(data.as_matrix(), target.as_matrix(), test_size = 0.20, random_state = 42)

# Working on loaded classifier model
classifier.fit(X_Train, Y_Train)
distances, neighbor = classifier.kneighbors(X_Test)

# Converting 2d numpy array to 1d list
book = list(itertools.chain.from_iterable(Y_Test.tolist()))
dict_books = {}

# Creating a dictionary of recommendations
for i in range(len(Y_Test)):
    # Remove symbols from title to be imported as json into Firebase database
    book[i] = re.sub(r'[^\w]', ' ', book[i])
    str = ""    
    temp_list = list(itertools.chain.from_iterable(Y_Train[neighbor[i]].tolist()))
    
    for j in range(len(temp_list)):
        str = str + re.sub(r'[^\w]', ' ', temp_list[j]) + ","
    
    dict_books[book[i]] = str[:-1]

# Write data to json
with open('recommendations.json', 'w') as f:
     json.dump(dict_books, f)

# Read data from json
# with open('recommendations.json', 'r') as f:
#    data = json.load(f)
