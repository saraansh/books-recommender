# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:01:54 2017

@author: Hopeless
"""

import pandas as pd
import numpy as np
import os
import ast

# Get the genre of a book
def getGenre(str):
    # Since we have the data in form of a dictionary
    # Parse the dictionary from string
    # Generate a list of genre (values only) from the dictionary
    return list(ast.literal_eval(str).values())

# Reading the dataset and assigning labels for each column
books = pd.read_csv(os.getcwd() + "/booksummaries.txt", sep='\t', names=['Wiki_ID', 'Freebase_ID', 'Tite', 'Author', 'Pub_Date', 'Genre', 'Plot'])

# Drop rows where subset = "null" and reset indexes
books.dropna(subset = ['Genre'], inplace = True)
books.reset_index(inplace = True, drop = True)

# Deleting unnecessary columns 
del books['Freebase_ID']
del books['Plot']
# del books['Author']

# Drop column with insufficient data
books.drop(['Pub_Date'], axis=1, inplace = True)

books['Genre'] = books['Genre'].apply(getGenre)

# Verifying Data
# Display the starting and ending five rows of dataset
books.head()
books.tail()

# Saving cleaned data
# books.to_csv("booksummaries_cleaned.txt", sep=";", index=None, encoding="utf-8")

