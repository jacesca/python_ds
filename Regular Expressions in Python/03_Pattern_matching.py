# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:18:02 2019

@author: jacqueline.cortez

Capítulo 3. Regular Expressions for Pattern Matching
Introduction:
    Time to discover the fundamental concepts of regular expressions! In this key chapter, 
    you will learn to understand the basic concepts of regular expression syntax. Using a real 
    dataset with tweets meant for sentiment analysis, you will learn how to apply pattern matching 
    using normal and special characters, and greedy and lazy quantifiers.
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
tema = '2. Are they bots?'; print("** %s\n" % tema)

sentiment_analysis = '@robot9! @robot4& I have a good feeling that the show isgoing to be amazing! @robot9$ @robot7%'
regex = r"@robot\d\W" # Write the regex
print(sentiment_analysis,'\n',re.findall(regex, sentiment_analysis)) # Find all matches of regex


print("****************************************************")
tema = '3. Find the numbers'; print("** %s\n" % tema)

sentiment_analysis = "Unfortunately one of those moments wasn't a giant squid monster. User_mentions:2, likes: 9, number of retweets: 7"
print(sentiment_analysis)
print(re.findall(r"User_mentions:\d", sentiment_analysis)) # Write a regex to obtain user mentions
print(re.findall(r"likes:\s\d", sentiment_analysis)) # Write a regex to obtain number of likes
print(re.findall(r"number\sof\sretweets:\s\d", sentiment_analysis)) # Write a regex to obtain number of retweets


print("****************************************************")
tema = '4. Match and split'; print("** %s\n" % tema)

sentiment_analysis = 'He#newHis%newTin love with$newPscrappy. #8break%He is&newYmissing him@newLalready'
regex_sentence = r"\W\dbreak\W" # Write a regex to match pattern separating sentences
sentiment_sub = re.sub(regex_sentence, " ", sentiment_analysis) # Replace the regex_sentence with a space
regex_words = r"\Wnew\w" # Write a regex to match pattern separating words
sentiment_final = re.sub(regex_words, ' ', sentiment_sub) # Replace the regex_words and print the result
print(sentiment_analysis,'\n',sentiment_final)


print("****************************************************")
tema = '6. Everything clean'; print("** %s\n" % tema)

sentiment_analysis =["Boredd. Colddd @blueKnight39 Internet keeps stuffing up. Save me! https://www.tellyourstory.com",
                     "I had a horrible nightmare last night @anitaLopez98 @MyredHat31 which affected my sleep, now I'm really tired",
                     "im lonely  keep me company @YourBestCompany! @foxRadio https://radio.foxnews.com 22 female, new york"]
for tweet in sentiment_analysis:
    print(tweet)
    print(re.findall(r"https?://[\w+.]*", tweet)) # Write regex to match http links and print out result
    print(re.findall(r"https?://.+", tweet)) # Write regex to match http links and print out result
    print(re.findall(r"@\w+", tweet),'\n')# Write regex to match user mentions and print out result


print("***************************************************")
tema = '7. Some time ago'; print("** %s\n" % tema)

sentiment_analysis =["I would like to apologize for the repeated Video Games Live related tweets. 32 minutes ago",
                     "@zaydia but i cant figure out how to get there / back / pay for a hotel 1st May 2019",
                     "FML: So much for seniority, bc of technological ineptness 23rd June 2018 17:54"]
for date in sentiment_analysis:
    print(date)
    print(re.findall(r"\d{1,2}\s\w+\sago", date))
    print(re.findall(r"\d{1,2}\w{2}\s\w+\s\d{4}", date))
    print(re.findall(r"\d{1,2}\w{2}\s\w+\s\d{4}\s\d{1,2}:\d{2}", date), '\n')


print("****************************************************")
tema = '8. Getting tokens'; print("** %s\n" % tema)

sentiment_analysis = 'ITS NOT ENOUGH TO SAY THAT IMISS U #MissYou #SoMuch #Friendship #Forever'
regex = r"#\w+" # Write a regex matching the hashtag pattern
no_hashtag = re.sub(regex, "", sentiment_analysis) # Replace the regex by an empty string
print(re.split(r"\s+", no_hashtag)) # Get tokens by splitting text

    
print("****************************************************")
tema = '9. Regex metacharacters'; print("** %s\n" % tema)

print(re.search(r'\d{4}',"4568 people attended."))
print(re.search(r'\d+',"There are 4568 people attended."))

m = re.search(r'\d+',"There are 4568 people attended.")
print(m.group(0), m.start(), m.end(), m.span(), m.re, m.string)

#Only match at the begining.
print(re.match(r'\d{4}',"4568 people attended."))
print(re.match(r'\d+',"There are 4568 people attended."))

m = re.match(r'\d{4}',"4568 people attended.")
print(m.group(0), m.start(), m.end(), m.span(), m.re, m.string)

#special characters
print(re.split(r'.\s', "I love the music of Mr.Go. However, the sound was too loud."))
print(re.split(r'.', "I love the music of Mr.Go. However, the sound was too loud."))
print(re.split(r'\s', "I love the music of Mr.Go. However, the sound was too loud."))
print(re.split(r'\.', "I love the music of Mr.Go. However, the sound was too loud."))

    
print("****************************************************")
tema = '10. Finding files'; print("** %s\n" % tema)

sentiment_analysis =["AIshadowhunters.txt aaaaand back to my literature review. At least i have a friendly cup of coffee to keep me company",
                     "ouMYTAXES.txt I am worried that I won't get my $900 even though I paid tax last year"]
regex = r"^[aeiouAEIOU]{2,3}.+txt" # Write a regex to match text file name
for text in sentiment_analysis:
    print(text)
    print(re.findall(regex, text)) # Find all matches of the regex
    print(re.sub(regex, '', text),'\n') # Replace all matches with empty string

    
print("****************************************************")
tema = '11. Give me your email'; print("** %s\n" % tema)

emails = ['n.john.smith@gmail.com', '87victory@hotmail.com', '!#mary-=@msca.net']
regex = r"[a-zA-Z0-9!#%&*$.]+@\w+\.com" # Write a regex to match a valid email address
print(emails)
for example in emails:
    if re.match(regex, example): # Match the regex to the string
        print("The email {email_example} is a valid email".format(email_example=example)) # Complete the format method to print out the result
    else:
        print("The email {email_example} is invalid".format(email_example=example))



print("****************************************************")
tema = '12. Invalid password'; print("** %s\n" % tema)

passwords = ['Apple34!rose', 'My87hou#4$', 'abc123']
regex = r"[a-zA-Z0-9*#$%!&.]{8,20}" # Write a regex to match a valid password
print(passwords)
for example in passwords: 
    if re.match(regex, example): # Scan the strings to find a match
      	print("The password {pass_example} is a valid password".format(pass_example=example)) # Complete the format method to print out the result
    else:
      	print("The password {pass_example} is invalid".format(pass_example=example))



print("****************************************************")
tema = '14. Understanding the difference'; print("** %s\n" % tema)

string = 'I want to see that <strong>amazing show</strong> again!'
string_notags = re.sub(r"<.+?>", "", string) # Write a regex to eliminate tags
print(string)
print(string_notags) # Print out the result


print("****************************************************")
tema = '15. Greedy matching'; print("** %s\n" % tema)

sentiment_analysis = 'Was intending to finish editing my 536-page novel manuscript tonight, but that will probably not happen. And only 12 pages are left '
print(sentiment_analysis)
print(re.findall(r"\d+?", sentiment_analysis))# Write a lazy regex expression 
print(re.findall(r"\d+", sentiment_analysis)) # Write a greedy regex expression 


print("****************************************************")
tema = '16. Lazy approach'; print("** %s\n" % tema)

sentiment_analysis = "Put vacation photos online (They were so cute) a few yrs ago. PC crashed, and now I forget the name of the site (I'm crying). "
print(sentiment_analysis)
print(re.findall(r"\(.+\)", sentiment_analysis)) # Write a greedy regex expression to match 
print(re.findall(r"\(.*?\)", sentiment_analysis))



print("****************************************************")
print("** END                                            **")
print("****************************************************")