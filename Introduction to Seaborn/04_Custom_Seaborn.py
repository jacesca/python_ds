# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:03:31 2019

@author: jacqueline.cortez

Capítulo 4. Customizing Seaborn Plots
Introduction:
    In this final chapter, you will learn how to add informative plot titles 
    and axis labels, which are one of the most important parts of any data 
    visualization! You will also learn how to customize the style of your 
    visualizations in order to more quickly orient your audience to the key
    takeaways. Then, you will put everything you have learned together for 
    the final exercises of the course!
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
#import statsmodels as sm

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

plt.rcParams = plt.rcParamsDefault
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.constrained_layout.h_pad'] = 0.09

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
survey_data["parents_advice"] = survey_data["Parents' advice"].replace([1, 2, 3, 4, 5, np.nan],['Never', 'Rarely','Sometimes', 'Often', 'Always', 'Never'])
survey_data['Number of Siblings'] = survey_data['Siblings'].apply(lambda x: '0' if ((x==0) | (np.isnan(x))) else ('1-2' if x<3 else '3+'))
survey_data['Interested in Pets'] = survey_data['Pets'].apply(lambda x: 'Yes' if (x>1) else 'No')
survey_data['Likes Techno'] = survey_data['Techno'].apply(lambda x: True if x > 3 else False)

file = "auto-mpg.csv"
print("Reading the data ({})...\n".format(file.upper()))
mpg = pd.read_csv(file, quotechar='"', skiprows=1,
                  names=['mpg','cylinders','displacement','horsepower','weight',
                         'acceleration','model_year','origin','name','color','size'])

mpg_mean = mpg.groupby(['model_year', 'origin']).mpg.mean().reset_index()

current_palette = sns.color_palette()
print(plt.style.available)

tema = '2. Changing style and palette'
print("****************************************************")
print("** %s\n" % tema)

# Set the color palette to "Purples"
sns.set_style("whitegrid")
sns.set_palette('Purples')

# Create a count plot of survey responses
category_order = ["Never", "Rarely", "Sometimes", "Often", "Always"]

g = sns.catplot(x="parents_advice", data=survey_data, kind="count", order=category_order)
g.fig.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('How often do you listen to your parents\' advice?') # Add the title
plt.show() # Show plot


# Change the color palette to "RdBu"
sns.set_palette("RdBu")

g = sns.catplot(x="parents_advice", data=survey_data, kind="count", order=category_order)
g.fig.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('How often do you listen to your parents\' advice?') # Add the title
plt.show() # Show plot

sns.set_style("white")
sns.set_palette(current_palette)


tema = '3. Changing the scale'
print("****************************************************")
print("** %s\n" % tema)

sns.set_context("poster")
g = sns.catplot(x="Number of Siblings", y="Loneliness", data=survey_data, kind="bar")
g.fig.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.title('Lonely feelings vs Siblings') # Add the title
plt.show() # Show plot

sns.set_context("paper")


tema = '4. Using a custom palette'
print("****************************************************")
print("** %s\n" % tema)

sns.set_style('darkgrid')
sns.set_palette(['#39A7D0', '#36ADA4']) # Set a custom color palette

# Create the box plot of age distribution by gender
g = sns.catplot(x="Gender", y="Age", data=survey_data, kind="box")
g.fig.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.title('Summary of the type of people answering this survey') # Add the title
plt.show() # Show plot

sns.set_style("white")
sns.set_palette(current_palette)


tema = '6. FacetGrids vs. AxesSubplots'
print("****************************************************")
print("** %s\n" % tema)
tema = '7. Adding a title to a FacetGrid object'
print("****************************************************")
print("** %s\n" % tema)

g = sns.relplot(x="weight", y="horsepower", data=mpg, kind="scatter", hue='origin')
g.fig.suptitle('{}\nCar Weight vs. Horsepower'.format(tema))
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.87, wspace=None, hspace=None)
plt.show() # Show plot
print('Plot type created: {}\n'.format(type(g)))


tema = '9. Adding a title and axis labels'
print("****************************************************")
print("** %s\n" % tema)

plt.figure()
g = sns.lineplot(x="model_year", y="mpg", data=mpg_mean, hue="origin")
g.set_title("Average MPG Over Time")
g.set(xlabel='Car Model Year', ylabel='Average MPG') # Add x-axis and y-axis labels
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.87, wspace=None, hspace=None)
plt.suptitle(tema)
plt.show() # Show plot
print('Plot type created: {}\n'.format(type(g)))


tema = '10. Rotating x-tick labels'
print("****************************************************")
print("** %s\n" % tema)

sns.catplot(x="origin", y="acceleration", data=mpg, kind="point", join=False, capsize=0.1)
plt.subplots_adjust(left=None, bottom=0.15, right=None, top=0.9, wspace=None, hspace=None)
plt.xticks(rotation=90)
plt.title('Cars Average Acceleration per Origin')
plt.suptitle(tema)
plt.show() # Show plot


tema = '12. Box plot with subgroups'
print("****************************************************")
print("** %s\n" % tema)

sns.set_palette('Blues') # Set palette to "Blues"

g = sns.catplot(x="Gender", y="Age", data=survey_data, kind="box", hue='Interested in Pets') # Adjust to add subgroups based on "Interested in Pets"
g.fig.suptitle('{}\nAge of Those Interested in Pets vs. Not'.format(tema)) # Set title to "Age of Those Interested in Pets vs. Not"
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.show() # Show plot

sns.set_palette(current_palette)



tema = '13. Bar plot with subgroups and subplots'
print("****************************************************")
print("** %s\n" % tema)

sns.set_style('dark') # Set the figure style to "dark"

g = sns.catplot(x="Village - town", y="Likes Techno", data=survey_data, kind="bar", col='Gender') # Adjust to add subplots per gender
g.fig.suptitle("{}\nPercentage of Young People Who Like Techno".format(tema)) # Add title and axis labels
g.set(xlabel="Location of Residence", ylabel="% Who Like Techno")
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None)
plt.show() # Show plot

sns.set_style("white")


print("****************************************************")
print("** END                                            **")
print("****************************************************")
