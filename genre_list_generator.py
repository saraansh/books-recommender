# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 02:44:38 2017

@author: Hopeless
"""

import pandas as pd
import numpy as np
import os
import ast


# Function to generate the genre_list
def extractGenre(str):
    genre_list = []
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
    return genre_list
                

# Reading the dataset
# No need to specify column headers
books = pd.read_csv(os.getcwd() + "/booksummaries_cleaned.txt", sep=';')

# Apply genre extraction
genre_list = books['Genre'].apply(extractGenre)

# Print the length of genre_list
print(len(genre_list))

# Saving cleaned data
# books.to_csv("booksummaries_cleaned.txt", sep=";", index=None, encoding="utf-8")