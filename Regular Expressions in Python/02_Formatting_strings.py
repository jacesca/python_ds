# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:18:02 2019

@author: jacqueline.cortez

Capítulo 2. Formatting Strings
Introduction:
    Following your journey, you will learn the main approaches that can be used to format or 
    interpolate strings in python using a dataset containing information scraped from the web. 
    You will explore the advantages and disadvantages of using positional formatting, embedding 
    expressing inside string constants, and using the Template class.
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
from datetime import datetime                                                        #For obteining today function
from string import Template                                                          #For working with string, regular expressions

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
tema = '2. Put it in order!'; print("** %s\n" % tema)

wikipedia_article = 'In computer science, artificial intelligence (AI), sometimes called machine intelligence, is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.'
my_list = []

first_pos = wikipedia_article[3:19].lower() # Assign the substrings to the variables
second_pos = wikipedia_article[21:44].lower()
my_list.append("The tool {} is used in {}.") # Define string with placeholders 
my_list.append("The tool {1} is used in {0}.") # Define string with rearranged placeholders
for my_string in my_list: # Use format to print strings
  	print(my_string.format(first_pos, second_pos))



print("****************************************************")
tema = '3. Calling by its name'; print("** %s\n" % tema)

courses = ['artificial intelligence', 'neural networks']
plan = {"field": courses[0],"tool": courses[1]} # Create a dictionary
my_message = "If you are interested in {plan[field]}, you can take the course related to {plan[tool]}." # Define string with placeholders
print(my_message.format(plan=plan)) # Use dictionary to replace placehoders



print("****************************************************")
tema = '4. What day is today?'; print("** %s\n" % tema)

get_date = datetime.today() # Assign date to get_date
message = "Good morning. Today is {today:%B %d, %Y}. It's {today:%H:%M} ... time to work!" # Add named placeholders with format specifiers
print(message.format(today=get_date)) # Format date



print("****************************************************")
tema = '5. Formatted string literal'; print("** %s\n" % tema)

name = 'word'
print(f"This {name} is just an example.")
print(f"Formating strings:       This {name!r} is just an example.")

number = 90.61890417471841
print(f"Formating floats:        This {number} is just an example.")
print(f"Formating floats:        This {number:.2f} is just an example.")
print(f"Formating floats:        This {number:.0f} is just an example.")

number = 90
print(f"Formmating integers:     This {number:3d} is just an example.")
print(f"Formmating integers:     This {number:1d} is just an example.")

my_date = datetime.today()
print(f"Working with dates:      This {my_date:%B %d, %Y} is just an example.")

family = {'dad':'Jose', 'mom':'Jacquie'}
print(f"Using dictionaries:      This {family['dad']} is just an example.")

a = 4
b = 7
print(f"Making operation inside: This {a-b} is just an example.")

def my_func(a, b):
    return a+b
print(f"Calling a function:      This {my_func(10,20)} is just an example.")



print("****************************************************")
tema = '6. Literally formatting'; print("** %s\n" % tema)

field1, field2, field3 = ('sexiest job', 'data is produced daily', 'Individuals')
fact1, fact2, fact3, fact4 = (21, 2500000000000000000, 72.41415415151, 1.09)
print(f"Data science is considered {field1!r} in the {fact1:d}st century") # Complete the f-string
print(f"About {fact2:e} of {field2} in the world") # Complete the f-string
print(f"{field3} create around {fact3:.2f}% of the data but only {fact4:.1f}% is analyzed") # Complete the f-string



print("****************************************************")
tema = '7. Make this function'; print("** %s\n" % tema)

number1, number2, string1 = (120, 7, 'httpswww.datacamp.com')
list_links = ['www.news.com', 'www.google.com', 'www.yahoo.com', 'www.bbc.com', 'www.msn.com', 'www.facebook.com', 'www.news.google.com']
print(f"{number1} tweets were downloaded in {number2} minutes indicating a speed of {number1/number2:.1f} tweets per min") # Include both variables and the result of dividing them 
print(f"{string1.replace('https', '')}") # Replace the substring https by an empty string
print(f"Only {len(list_links)*100/120:.2f}% of the posts contain links") # Divide the length of list by 120 rounded to two decimals



print("****************************************************")
tema = '8. On time'; print("** %s\n" % tema)

east = {'date': datetime(2007, 4, 20, 0, 0), 'price': 1232443}
west = {'date': datetime(2006, 5, 26, 0, 0), 'price': 1432673}
print(f"The price for a house in the east neighborhood was ${east['price']:,.2f} in {east['date']:%m-%d-%Y}") # Access values of date and price in east dictionary
print(f"The price for a house in the west neighborhood was ${west['price']:,.2f} in {west['date']:%m-%d-%Y}.") # Access values of date and price in west dictionary


    
print("****************************************************")
tema = '10. Preparing a report'; print("** %s\n" % tema)

tool1, tool2, tool3 = ('Natural Language Toolkit', 'TextBlob', 'Gensim')
description1, description2, description3 = ('suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. It was developed by Steven Bird and Edward Loper in the Department of Computer and Information Science at the University of Pennsylvania.',
                                            'Python library for processing textual data. It provides a simple API for diving into common natural language processing tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.',
                                            'Gensim is a robust open-source vector space modeling and topic modeling toolkit implemented in Python. It uses NumPy, SciPy and optionally Cython for performance. Gensim is specifically designed to handle large text collections, using data streaming and efficient incremental algorithms, which differentiates it from most other scientific software packages that only target batch and in-memory processing.')
wikipedia = Template("$tool is a $description") # Create a template
print(wikipedia.substitute(tool=tool1, description=description1)) # Substitute variables in template
print(wikipedia.substitute(tool=tool2, description=description2))
print(wikipedia.substitute(tool=tool3, description=description3))


    
print("****************************************************")
tema = '11. Identifying prices'; print("** %s\n" % tema)

tools = ['Natural Language Toolkit', '20', 'month']
our_tool, our_fee, our_pay = tools[0], tools[1], tools[2] # Select variables
course = Template("We are offering a 3-month beginner course on $tool just for $$ $fee ${pay}ly") # Create template
print(course.substitute(tool=our_tool, fee=our_fee, pay=our_pay)) # Substitute identifiers with three variables



print("****************************************************")
tema = '12. Playing safe'; print("** %s\n" % tema)

answers = {'answer1': 'I really like the app. But there are some features that can be improved'}
the_answers = Template("Check your answer 1: $answer1, and your answer 2: $answer2") # Complete template string using identifiers
# Use substitute to replace identifiers
try:
    print(the_answers.substitute(answers))
except KeyError:
    print("Missing information")
# Use safe_substitute to replace identifiers
try:
    print(the_answers.safe_substitute(answers))
except KeyError:
    print("Missing information")



print("****************************************************")
print("** END                                            **")
print("****************************************************")