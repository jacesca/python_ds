# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:22:49 2019

@author: jacqueline.cortez

Capítulo 3: Visualizing a Categorical and a Quantitative Variable
Introduction:
    Categorical variables are present in nearly every dataset, but 
    they are especially prominent in survey data. In this chapter, you 
    will learn how to create and customize categorical plots such as box 
    plots, bar plots, count plots, and point plots. Along the way, you 
    will explore survey data from young people about their interests, 
    students about their study habits, and adult men about their feelings 
    about masculinity.
"""


# Import packages
import pandas as pd
import numpy as np
#import tabula 
#import math
import matplotlib.pyplot as plt
import seaborn as sns
#import scipy.stats as stats
#import random
#import calendar

#from pandas.plotting import register_matplotlib_converters #for conversion as datetime index in x-axis
#from math import radians
#from functools import reduce#import pandas as pd
#from pandas.api.types import CategoricalDtype #For categorical data
#from glob import glob
#from bokeh.io import output_file, show
#from bokeh.plotting import figure

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#register_matplotlib_converters() #Require to explicitly register matplotlib converters.

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("** Getting the data for this program\n")

#file = 'http://assets.datacamp.com/production/repositories/3996/datasets/ab13162732ae9ca1a9a27e2efd3da923ed6a4e7b/young-people-survey-responses.csv'
file = 'Survey_spiderscare.csv'
print("Reading the data ({})...\n".format(file.upper()))
survey_data = pd.read_csv(file)
survey_data['age category'] = survey_data['Age'].apply(lambda x: 'Less than 21' if x < 21 else '21 +')
survey_data['Interested in Math'] = survey_data['Mathematics'].apply(lambda x: True if x > 3 else False)

#file = 'https://assets.datacamp.com/production/repositories/3996/datasets/61e08004fef1a1b02b62620e3cd2533834239c90/student-alcohol-consumption.csv'
file = 'student_data.csv'
print("Reading the data ({})...\n".format(file.upper()))
student_data = pd.read_csv(file)


tema = '2. Count plots'
print("****************************************************")
print("** %s\n" % tema)

# Create column subplots based on age category
g = sns.catplot(y="Internet usage", data=survey_data, kind="count", 
                col='age category')
g.fig.suptitle('{}\nRelationship between study time and final grade (scatter)'.format(tema))
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show plot

tema = '3. Bar plots with percentages'
print("****************************************************")
print("** %s\n" % tema)

# Create column subplots based on age category
sns.catplot(kind='bar', x='Gender', y='Interested in Math', data=survey_data)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('Interest in Math by gender') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '4. Customizing bar plots'
print("****************************************************")
print("** %s\n" % tema)

# Create column subplots based on age category
sns.catplot(x="study_time", y="G3", data=student_data, kind="bar",
            order=["<2 hours", "2 to 5 hours", "5 to 10 hours", ">10 hours"], 
            ci=None)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('Study time vs Final Grade') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '6. Create and interpret a box plot'
print("****************************************************")
print("** %s\n" % tema)

# Specify the category ordering
study_time_order = ["<2 hours", "2 to 5 hours", 
                    "5 to 10 hours", ">10 hours"]

# Specify the category ordering
sns.catplot(kind='box', x='study_time', y='G3', data=student_data, order=study_time_order)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('Study time vs Final Grade') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '7. Omitting outliers'
print("****************************************************")
print("** %s\n" % tema)

# Create a box plot with subgroups and omit the outliers
sns.catplot(kind='box', x='internet', y='G3', data=student_data, hue='location', sym='')
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('Relation between Internet and Final grade') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '8. Adjusting the whiskers'
print("****************************************************")
print("** %s\n" % tema)

# Set the whiskers at the min and max values
sns.catplot(x="romantic", y="G3", data=student_data, kind="box", whis=[0,100])
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('Relation between Final grade and being in a romantic relationship') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '10. Customizing point plots'
print("****************************************************")
print("** %s\n" % tema)

# Remove the lines joining the points
sns.catplot(x="famrel", y="absences", data=student_data, kind="point", capsize=0.2)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('Does the quality of the student\'s family relationship influence the number of absences the student has in school?') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


# Remove the lines joining the points
sns.catplot(x="famrel", y="absences", data=student_data, kind="point", capsize=0.2, join=False)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('Does the quality of the student\'s family relationship influence the number of absences the student has in school?') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


tema = '11. Point plots with subgroups'
print("****************************************************")
print("** %s\n" % tema)

# Plot the median number of absences instead of the mean
sns.catplot(x="romantic", y="absences", data=student_data, 
            kind="point", hue="school", 
            ci=None, estimator=np.median)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('Is being in a romantic relationship associated with higher or lower school attendance? ') # Add the title
plt.suptitle(tema)
plt.show() # Show plot


print("****************************************************")
print("** END                                            **")
print("****************************************************")
