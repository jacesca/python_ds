# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:18:02 2019

@author: jacqueline.cortez

Capítulo 4. Advanced Regular Expression Concepts
Introduction:
    In the last step of your journey, you will learn more complex methods of pattern matching using parentheses to group 
    strings together or to match the same text as matched previously. Also, you will get an idea of how you can look around 
    expressions.
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
import re                             #For regular expressions

#from pandas.plotting import register_matplotlib_converters                          #For conversion as datetime index in x-axis
#from math import radian                                                             #For accessing a specific math operations
#from functools import reduce                                                        #For accessing to a high order functions (functions or operators that return functions)
#from pandas.api.types import CategoricalDtype                                       #For categorical data
#from glob import glob                                                               #For using with pathnames matching
#from datetime import datetime                                                        #For obteining today function
#from string import Template                                                          #For working with string, regular expressions

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
tema = '2. Try another name'; print("** %s\n" % tema)

sentiment_analysis = ['Just got ur newsletter, those fares really are unbelievable. Write to statravelAU@gmail.com or statravelpo@hotmail.com. They have amazing prices',
                      'I should have paid more attention when we covered photoshop in my webpage design class in undergrad. Contact me Hollywoodheat34@msn.net.',
                      'hey missed ya at the meeting. Read your email! msdrama098@hotmail.com']
regex_email = r"([a-zA-Z0-9]+)@\S+" # Write a regex that matches email
for tweet in sentiment_analysis:
    email_matched = re.findall(regex_email, tweet) # Find all matches of regex in each tweet
    print(tweet,"\nLists of users found in this tweet: {}".format(email_matched), '\n') # Complete the format method to print the results
    

print("****************************************************")
tema = '3. Flying home'; print("** %s\n" % tema)

flight = 'Subject: You are now ready to fly. Here you have your boarding pass IB3723 AMS-MAD 06OCT'
regex = r"([A-Z]{2})(\d{4})\s([A-Z]{3})-([A-Z]{3})\s(\d{2}[A-Z]{3})" # Write regex to capture information of the flight
flight_matches = re.findall(regex, flight) # Find all matches of the flight information
print(flight)
print(flight_matches)
print("Airline: {} Flight number: {}".format(flight_matches[0][0], flight_matches[0][1])) #Print the matches
print("Departure: {} Destination: {}".format(flight_matches[0][2], flight_matches[0][3]))
print("Date: {}".format(flight_matches[0][4]))



print("****************************************************")
tema = '5. Love it!'; print("** %s\n" % tema)

sentiment_analysis = ['I totally love the concert The Book of Souls World Tour. It kinda amazing!',
                      'I enjoy the movie Wreck-It Ralph. I watched with my boyfriend.',
                      "I still like the movie Wish Upon a Star. Too bad Disney doesn't show it anymore."]
regex_positive = r"(love|like|enjoy).+?(movie|concert)\s(.+?)\." # Write a regex that matches sentences with the optional words
for tweet in sentiment_analysis:
    print(tweet)
    positive_matches = re.findall(regex_positive, tweet) # Find all matches of regex in tweet
    print("Positive comments found {}.\n".format(positive_matches)) # Complete format to print out the results
    

print("****************************************************")
tema = '6. Ugh! Not for me!'; print("** %s\n" % tema)

sentiment_analysis = ['That was horrible! I really dislike the movie The cabin and the ant. So boring.',
                      "I disapprove the movie Honest with you. It's full of cliches.",
                      'I dislike very much the concert After twelve Tour. The sound was horrible.']
regex_negative = r"(hate|dislike|disapprove).+?(?:movie|concert)\s(.+?)\." # Write a regex that matches sentences with the optional words
for tweet in sentiment_analysis:
    print(tweet)
    negative_matches = re.findall(regex_negative, tweet) # Find all matches of regex in tweet
    print("Negative comments found {}.\n".format(negative_matches)) # Complete format to print out the results
    

print("****************************************************")
tema = '8. Parsing PDF files'; print("** %s\n" % tema)

contract = 'Provider will invoice Client for Services performed within 30 days of performance.  Client will pay Provider as set forth in each Statement of Work within 30 days of receipt and acceptance of such invoice. It is understood that payments to Provider for services rendered shall be made in full as agreed, without any deductions for taxes of any kind whatsoever, in conformity with Provider’s status as an independent contractor. Signed on 03/25/2001.'
print(contract, "\n")
regex_dates = r"Signed\son\s(\d{2})/(\d{2})/(\d{4})" # Write regex and scan contract to capture the dates described
dates = re.search(regex_dates, contract)
signature = {"day": dates.group(2), "month": dates.group(1), "year": dates.group(3)} # Assign to each key the corresponding match
print("Our first contract is dated back to {data[year]}. Particularly, the day {data[day]} of the month {data[month]}.".format(data=signature)) # Complete the format method to print-out


print("****************************************************")
tema = '9. Close the tag, please!'; print("** %s\n" % tema)

html_tags = ['<body>Welcome to our course! It would be an awesome experience</body>',
             '<article>To be a data scientist, you need to have knowledge in statistics and mathematics</article>',
             '<nav>About me Links Contact me!']
for string in html_tags:
    match_tag =  re.match(r"<(\w+)>.*?</\1>", string) # Complete the regex and find if it matches a closed HTML tags
    if match_tag:
        print("Your tag {} is closed".format(match_tag.group(1))) # If it matches print the first group capture
    else:
        notmatch_tag = re.match(r"<(\w+)>", string) # If it doesn't match capture only the tag 
        print("Close your {} tag!".format(notmatch_tag.group(1))) # Print the first group capture



print("****************************************************")
tema = '10. Reeepeated characters'; print("** %s\n" % tema)

sentiment_analysis = ['@marykatherine_q i know! I heard it this morning and wondered the same thing. Moscooooooow is so behind the times',
                      'Staying at a friends house...neighborrrrrrrs are so loud-having a party',
                      'Just woke up an already have read some e-mail']
regex_elongated = r"\w+(\w)+\1+\w*" # Complete the regex to match an elongated word
for tweet in sentiment_analysis:
    match_elongated = re.search(regex_elongated, tweet) # Find if there is a match in each tweet 
    if match_elongated:
        elongated_word = match_elongated.group(0) # Assign the captured group zero 
        print("Elongated word found: {word}".format(word=elongated_word)) # Complete the format method to print the word
    else:
        print("No elongated word found") 


print("****************************************************")
tema = '11. Lookaround'; print("** %s\n" % tema)

my_text = 'tweets.txt transferred, mypass.txt transferred, keywords.txt error'
print(my_text)
print("Positive look-ahead by transferred: {}.\n".format(re.findall(r"\w+\.txt(?=\stransferred)", my_text)))

my_text = 'tweets.txt transferred, mypass.txt transferred, keywords.txt error'
print(my_text)
print("Negative look-ahead by transferred: {}.\n".format(re.findall(r"\w+\.txt(?!\stransferred)", my_text)))

my_text = 'My white cat sat at the table. However, my brown dog was lying on the couch.'
print(my_text)
print("Positive look-behind by brown: {}.\n".format(re.findall(r"(?<=brown\s)(cat|dog)", my_text)))

my_text = 'My white cat sat at the table. However, my brown dog was lying on the couch.'
print(my_text)
print("Negative look-behind by brown: {}.\n".format(re.findall(r"(?<!brown\s)(cat|dog)", my_text)))


print("****************************************************")
tema = '12. Surrounding words'; print("** %s\n" % tema)

sentiment_analysis = 'You need excellent python skills to be a data scientist. Must be! Excellent python'
print(sentiment_analysis)
look_ahead = re.findall(r"\w+(?=\spython)", sentiment_analysis) # Positive lookahead
print("Look ahead by python: ", look_ahead) # Print out
look_behind = re.findall(r"(?<=[pP]ython\s)\w+", sentiment_analysis) # Positive lookbehind
print("Look behind by python or Python", look_behind) # Print out


print("****************************************************")
tema = '13. Filtering phone numbers'; print("** %s\n" % tema)

cellphones = ['4564-646464-01', '345-5785-544245', '6476-579052-01']
print(cellphones,'\n')
print("Not preceeding by area code:")
for phone in cellphones:
    number = re.findall(r"(?<!\d{3}-)\d{4}-\d{6}-\d{2}", phone) # Get all phone numbers not preceded by area code
    print(number)
print("\n")
print("Not preceeding by optional extension:")
for phone in cellphones:
    number = re.findall(r"\d{3}-\d{4}-\d{6}(?!-\d{2})", phone) # Get all phone numbers not followed by optional extension
    print(number)


print("****************************************************")
print("** END                                            **")
print("****************************************************")