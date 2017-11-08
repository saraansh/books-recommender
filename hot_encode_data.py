# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 02:44:38 2017

@author: Hopeless
"""

import pandas as pd
import numpy as np
import os
import ast

genre_list = []

# Reading the dataset
# No need to specify column headers
books = pd.read_csv(os.getcwd() + "/booksummaries_cleaned.txt", sep=';')

# Function to generate the genre_list
def extractGenre(str):
    # Check for errors if any
    try:
        temp_list = ast.literal_eval(str)
        for new in temp_list:
            flag = True
            for old in genre_list:
                if (new.lower() == old.lower()):
                    flag = False
                    break
            if (flag):
                genre_list.append(new)
    except Exception:
        print(str)

# Function to hot encode data
def setGenre():
    for pos in range(len(books['Title'])):
        # Extract list of genre for each book
        temp_list = ast.literal_eval(books.iloc[pos, books.columns.get_loc('Genre')])
        for genre in temp_list:
            # Set the value of cell (pos,'genre')=1 when match is found
            books.iloc[pos, books.columns.get_loc(genre)] = 1


# Apply genre extraction
books['Genre'].apply(extractGenre)

# Print the length of genre_list
# print(len(genre_list))

idx = 4
# Adding columns for all genre for hot encoding
for genre in genre_list:
    books.insert(loc=idx, column=genre, value=np.zeros(len(books['Title'])))
    idx += 1
    
setGenre()
    
# Saving hot encoded data
books.to_csv("booksummaries_hot_encoded.txt", sep=";", index=None, encoding="utf-8")