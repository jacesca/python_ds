# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:55:03 2019

@author: jacqueline.cortez

Introduction:
    Here you will learn how to write your very own functions. 
    In this Chapter, you'll learn how to write simple functions, 
    as well as functions that accept multiple arguments and return
    multiple values. You'll also have the opportunity to apply these 
    newfound skills to questions that commonly arise in Data Science 
    contexts.
"""

# Import the pandas library as pd
import pandas as pd
#import matplotlib.pyplot as plt
#from pandas.api.types import CategoricalDtype #For categorical data

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("Write a simple function\n")

# Define the function shout
def shout():
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word="congratulations"+"!!!"

    # Print shout_word
    print(shout_word)

# Call shout
shout()

print("\n****************************************************")
print("Single-parameter functions\n")

# Define shout with the parameter, word
def shout(word):
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'

    # Print shout_word
    print(shout_word)

# Call shout with the string 'congratulations'
shout("congratulations")

print("\n****************************************************")
print("Functions that return single values\n")

# Define shout with the parameter, word
def shout(word):
    """Return a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word=word+"!!!"

    # Replace print with return
    return shout_word

# Pass 'congratulations' to shout: yell
yell=shout("congratulations")

# Print yell
print(yell)

print("\n****************************************************")
print("Functions with multiple parameters\n")

# Define shout with parameters word1 and word2
def shout(word1, word2):
    """Concatenate strings with three exclamation marks"""
    # Concatenate word1 with '!!!': shout1
    shout1=word1+"!!!"
    
    # Concatenate word2 with '!!!': shout2
    shout2=word2+"!!!"
    
    # Concatenate shout1 with shout2: new_shout
    new_shout=shout1 + shout2

    # Return new_shout
    return new_shout

# Pass 'congratulations' and 'you' to shout(): yell
yell=shout("congratulations","you")

# Print yell
print(yell)

print("\n****************************************************")
print("A brief introduction to tuples\n")

nums = (3, 4, 6)

# Unpack nums into num1, num2, and num3
num1, num2, num3 = nums

# Construct even_nums
even_nums = (2,4,6)

print(num1)
print(num2)
print(num3)
print(even_nums)

print("\n****************************************************")
print("Functions that return multiple values\n")

# Define shout_all with parameters word1 and word2
def shout_all(word1, word2):
    
    # Concatenate word1 with '!!!': shout1
    shout1=word1+"!!!"
    
    # Concatenate word2 with '!!!': shout2
    shout2=word2+"!!!"
    
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words=(shout1, shout2)

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1,yell2=shout_all("congratulations","you")

# Print yell1 and yell2
print(yell1)
print(yell2)

print("\n****************************************************")
print("Bringing it all together (1)\n")

# Import Twitter data as DataFrame: df
tweets_df = pd.read_csv("tweets.csv", sep = ";", encoding="ISO-8859-1")

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = tweets_df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry]=langs_count[entry]+1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry]=1

# Print the populated dictionary
print(langs_count)

print("\n****************************************************")
print("Bringing it all together (2)\n")

# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of 
    occurrences as value for each key."""

    # Initialize an empty dictionary: langs_count
    langs_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over lang column in DataFrame
    for entry in col:

        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry]=langs_count[entry]+1
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry]=1

    # Return the langs_count dictionary
    return langs_count

# Call count_entries(): result
result = count_entries(tweets_df,"lang")

# Print the result
print(result)

print("****************************************************")
print("** END                                            **")
print("****************************************************")