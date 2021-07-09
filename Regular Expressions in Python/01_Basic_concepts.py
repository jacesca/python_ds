# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:18:02 2019

@author: jacqueline.cortez

Capítulo 1. Basic Concepts of String Manipulation
Introduction:
    Start your journey into the regular expression world! From slicing and concatenating, 
    adjusting the case, removing spaces, to finding and replacing strings. You will learn 
    how to master basic operation for string manipulation using a movie review dataset.
"""

# Import packages
#import pandas as pd                   #For loading tabular data
#import numpy as np                    #For making operations in lists
#import matplotlib as mpl              #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.pyplot as plt       #For creating charts
#import seaborn as sns                 #For visualizing data
#import scipy.stats as stats          #For accesign to a vary of statistics functiosn
#import statsmodels as sm             #For stimations in differents statistical models
#import scykit-learn                  #For performing machine learning  
#import tabula                        #For extracting tables from pdf
#import nltk                          #For working with text data
#import math                          #For accesing to a complex math operations
#import random                        #For generating random numbers
#import calendar                      #For accesing to a vary of calendar operations

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching

#from bokeh.io import curdoc, output_file, show                                      #For interacting visualizations
#from bokeh.plotting import figure, ColumnDataSource                                 #For interacting visualizations
#from bokeh.layouts import row, widgetbox, column, gridplot                          #For interacting visualizations
#from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper        #For interacting visualizations
#from bokeh.models import Slider, Select, Button, CheckboxGroup, RadioGroup, Toggle  #For interacting visualizations
#from bokeh.models.widgets import Tabs, Panel                                        #For interacting visualizations
#from bokeh.palettes import Spectral6                                                #For interacting visualizations

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#register_matplotlib_converters() #Require to explicitly register matplotlib converters.

#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09

#Setting the numpy options
#np.set_printoptions(precision=3) #precision set the precision of the output:
#np.set_printoptions(suppress=True) #suppress suppresses the use of scientific notation for small numbers

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Getting the data for this program\n")

print("****************************************************")
tema = '2. First day!'; print("** %s\n" % tema)

movie = 'fox and kelley soon become bitter rivals because the new fox books store is opening up right across the block from the small business .'
length_string = len(movie) # Find characters in movie variable
to_string = str(length_string) # Convert to string
statement = "Number of characters in this review:" # Predefined variable
print(statement + ' ' + to_string) # Concatenate strings and print result



print("****************************************************")
tema = '3. Artificial reviews'; print("** %s\n" % tema)

movie1 = 'the most significant tension of _election_ is the potential relationship between a teacher and his student .'
movie2 = 'the most significant tension of _rushmore_ is the potential relationship between a teacher and his student .'
first_part = movie1[:32] # Select the first 32 characters of movie1
last_part = movie1[42:] # Select from 43rd character to the end of movie1
middle_part = movie2[32:42] # Select from 33rd to the 42nd character
print(first_part+middle_part+last_part) # Print concatenation and movie2 variable
print(movie2)



print("****************************************************")
tema = '4. Palindromes'; print("** %s\n" % tema)

movie = 'oh my God! desserts I stressed was an ugly movie'
movie_title = movie[11:30] # Get the word
palindrome = movie_title[::-1] # Obtain the palindrome
if movie_title == palindrome: # Print the word if it's a palindrome
	print(movie_title)



print("****************************************************")
tema = '6. Normalizing reviews'; print("** %s\n" % tema)

movie = '$I supposed that coming from MTV Films I should expect no less$'
movie_lower = movie.lower() # Convert to lowercase and print the result
print(movie_lower)
movie_no_space = movie_lower.strip("$") # Remove whitespaces and print the result
print(movie_no_space)
movie_split = movie_no_space.split() # Split the string into substrings and print the result
print(movie_split)
word_root = movie_split[1][0:-1] # Select root word and print the result
print(word_root)



print("****************************************************")
tema = '7. Time to join!'; print("** %s\n" % tema)

movie = 'the film,however,is all good<\\i>'
movie_tag = movie.rstrip("<\i>") # Remove tags happening at the end and print results
print(movie_tag)
movie_no_comma = movie_tag.split(",") # Split the string using commas and print results
print(movie_no_comma)
movie_join = " ".join(movie_no_comma) # Join back together and print results
print(movie_join)


print("****************************************************")
tema = '8. Split lines or split the line?'; print("** %s\n" % tema)

file = 'mtv films election, a high school comedy, is a current example\nfrom there, director steven spielberg wastes no time, taking us into the water on a midnight swim'

file_split = file.splitlines() # Split string at line boundaries
print(file_split) # Print file_split
for substring in file_split: # Complete for-loop to split by commas
    substring_split = substring.split(',')
    print(substring_split)
    
    
    
print("****************************************************")
tema = '10. '; print("** %s\n" % tema)

movies = ["200    it's clear that he's passionate about his beliefs , and that he's not just a punk looking for an excuse to beat people up .",
          "201    I believe you I always said that the actor actor actor is amazing in every movie he has played",
          "202    it's astonishing how frightening the actor actor norton looks with a shaved head and a swastika on his chest."]
for movie in movies:
    print(movie)
    if movie.find('actor') == -1: # Find if actor occurrs between 37 and 41
        print("Word not found")
    elif movie.count('actor') == 2:  # Count occurrences and replace two by one
        print(movie.replace('actor actor', 'actor'))
    else:
        print(movie.replace('actor actor actor', 'actor')) # Replace three occurrences by one
    

    
print("****************************************************")
tema = '11. Where\'s the word?'; print("** %s\n" % tema)

movies = ["137    heck , jackie doesn't even have enough money for a haircut , looks like , much less a personal hairstylist .",
          "138    in condor , chan plays the same character he's always played , himself , a mixture of bruce lee and tim allen , a master of both kung-fu and slapstick-fu ."]
print('Using find:')  
for movie in movies:
  print(movie.find('money', 12, 51)) # Find the first occurrence of word
print('Using index:')  
for movie in movies:
  try:
  	print(movie.index('money', 12, 51)) # Find the first occurrence of word
  except ValueError:
    print("substring not found")



print("****************************************************")
tema = '12. Replacing negations'; print("** %s\n" % tema)

movies = "the rest of the story isn't important because all it does is serve as a mere backdrop for the two stars to share the screen ."
print(movies)
movies_no_negation = movies.replace("isn't", "is") # Replace negations 
movies_antonym = movies_no_negation.replace("important", "insignificant") # Replace important
print(movies_antonym) # Print out



print("****************************************************")
print("** END                                            **")
print("****************************************************")