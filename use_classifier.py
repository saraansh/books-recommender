# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 04:05:52 2017

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

# Load Classifier
classifier_new = open(os.getcwd() + "/knn_classifer.pickle","rb")
classifier = pickle.load(classifier_new)


# Code in the work to be done with the classifier
# For example you can use a new X_test and Y_test
# and use this to get score or accuracy of classifier
# classifier.score(X_test, Y_test)
# or you could use it for generating new recommendations
# based on the new inputs like in the previous example
# distances, neighbor = classifier.neighbors(X_tests)


classifier_new.close()