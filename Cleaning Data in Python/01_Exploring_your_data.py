# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:52:04 2020

@author: jacesca@gmail.com
Chapter1 - Exploring your data:
    Say you've just gotten your hands on a brand new dataset and are itching to 
    start exploring it. But where do you begin, and how can you be sure your 
    dataset is clean? This chapter will introduce you to data cleaning in Python. 
    You'll learn how to explore your data with an eye for diagnosing issues such 
    as outliers, missing values, and duplicate rows.
Source: https://learn.datacamp.com/courses/cleaning-data-in-python
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import matplotlib.pyplot as plt
import pandas as pd


print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

# Global params
suptitle_param = dict(color='darkblue', fontsize=12)
title_param = {'color': 'darkred', 'fontsize': 14}

plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 8, 'font.size': 8})
    

# Reading data
female_education = pd.read_csv('female_education.csv', sep=';')
job_app = pd.read_csv('dob_job_application_filings_subset.csv', low_memory=False)


print("****************************************************")
topic = "1. Diagnose data for cleaning"; print("** %s" % topic)

print(female_education.head(),"\n\n")

print(female_education.info(),'\n\n')

print("Columns: ",female_education.columns,'\n\n')

print("Shape: ",female_education.shape,'\n')

print(female_education.Continent.value_counts(dropna=False),'\n\n')
print(female_education['Country '].value_counts(dropna=False),'\n\n')


print("****************************************************")
topic = "2. Loading and viewing your data"; print("** %s\n" % topic)

"""
Loading and viewing your data
In this chapter, you're going to look at a subset of the Department of Buildings 
Job Application Filings dataset from the NYC Open Data portal. This dataset 
consists of job applications filed on January 22, 2017.
Your first task is to load this dataset into a DataFrame and then inspect it 
using the .head() and .tail() methods. However, you'll find out very quickly 
that the printed results don't allow you to see everything you need, since there 
are too many columns. Therefore, you need to look at the data in another way.
The .shape and .columns attributes let you see the shape of the DataFrame and 
obtain a list of its columns. From here, you can see which columns are relevant 
to the questions you'd like to ask of the data. To this end, a new DataFrame, 
df_subset, consisting only of these relevant columns, has been pre-loaded. This 
is the DataFrame you'll work with in the rest of the chapter.
Get acquainted with the dataset now by exploring it with pandas! This initial 
exploratory analysis is a crucial first step of data cleaning.
"""

print(job_app.head(),'\n\n')


print("****************************************************")
topic = "3. Further diagnosis"; print("** %s\n" % topic)

"""
Further diagnosis
In the previous exercise, you identified some potentially unclean or missing data. 
Now, you'll continue to diagnose your data with the very useful .info() method.
The .info() method provides important information about a DataFrame, such as the 
number of rows, number of columns, number of non-missing values in each column, 
and the data type stored in each column. This is the kind of information that 
will allow you to confirm whether the 'Initial Cost' and 'Total Est. Fee' columns 
are numeric or strings. From the results, you'll also be able to see whether or 
not all columns have complete data in them.
The full DataFrame df and the subset DataFrame df_subset have been pre-loaded. 
Your task is to use the .info() method on these and analyze the results.
"""
print(job_app.info(),'\n\n')
print("Columns: ",job_app.columns,'\n\n')


print("****************************************************")
topic = "4. Exploratory data analysis"; print("** %s\n" % topic)

print(female_education.describe(),'\n\n')

print("****************************************************")
topic = "5. Calculating summary statistics"; print("** %s\n" % topic)

"""
Calculating summary statistics
You'll now use the .describe() method to calculate summary statistics of your data.
In this exercise, an adapted DataFrame has been prepared for you to inspect, with 
fewer columns to increase readability in the IPython Shell.
This adapted DataFrame has been pre-loaded as df. Your job is to use the .describe() 
method on it in the IPython Shell and select the statement below that is False.
"""
print(job_app.describe(),'\n\n')
print(job_app['Proposed Zoning Sqft'].mean(),'\n\n')

print("****************************************************")
topic = "6. Frequency counts for categorical data"; print("** %s\n" % topic)

"""
Frequency counts for categorical data
As you've seen, .describe() can only be used on numeric columns. So how can you 
diagnose data issues when you have categorical data? One way is by using the 
.value_counts() method, which returns the frequency counts for each unique value 
in a column!
This method also has an optional parameter called dropna which is True by default. 
What this means is if you have missing data in a column, it will not give a frequency 
count of them. You want to set the dropna column to False so if there are missing values 
in a column, it will give you the frequency counts.
In this exercise, you're going to look at the 'Borough', 'State', and 'Site Fill' 
columns to make sure all the values in there are valid. When looking at the output, 
do a sanity check: Are all values in the 'State' column from NY, for example? Since 
the dataset consists of applications filed in NY, you would expect this to be the case.
"""
print(job_app['Zoning Dist2'].value_counts(),'\n\n')
print(job_app['Zoning Dist2'].value_counts(dropna=False),'\n\n')


print("****************************************************")
topic = "7. Visual exploratory data analysis"; print("** %s\n" % topic)

fig, axis = plt.subplots(1,3, figsize=(10,4))

ax = axis[0]
female_education['population'].plot(kind='hist', rwidth=.9, ax=ax)
ax.set_title('EDA - Hist Populattion', **title_param)
ax.set_xlabel('Population')

ax = axis[1]
female_education.boxplot(column='population', ax=ax)
ax.set_title('EDA - Boxplot', **title_param)

ax = axis[2]
female_education.boxplot(column='population', by='Continent', ax=ax)
ax.set_title('EDA - Boxplot Grouped by Continent', **title_param)
ax.set_ylabel('Population')

# Display the graph
fig.suptitle(topic, **suptitle_param)
plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=.5, hspace=None); #To set the margins 
plt.show()

print(female_education[female_education.population > 1e+9],'\n\n')


print("****************************************************")
topic = "8. Visualizing single variables with histograms"; print("** %s\n" % topic)

"""
Visualizing single variables with histograms
Up until now, you've been looking at descriptive statistics of your data. One of the best 
ways to confirm what the numbers are telling you is to plot and visualize the data.
You'll start by visualizing single variables using a histogram for numeric values. The column 
you will work on in this exercise is 'Existing Zoning Sqft'.
The .plot() method allows you to create a plot of each column of a DataFrame. The kind 
parameter allows you to specify the type of plot to use - kind='hist', for example, plots 
a histogram.
In the IPython Shell, begin by computing summary statistics for the 'Existing Zoning Sqft' 
column using the .describe() method. You'll notice that there are extremely large differences 
between the min and max values, and the plot will need to be adjusted accordingly. In such 
cases, it's good to look at the plot on a log scale. The keyword arguments logx=True or 
logy=True can be passed in to .plot() depending on which axis you want to rescale.
Finally, note that Python will render a plot such that the axis will hold all the information. 
That is, if you end up with large amounts of whitespace in your plot, it indicates counts or 
values too small to render.
"""
# Describe the column
print(job_app['Existing Zoning Sqft'].describe(),'\n\n')
print(job_app['Existing Zoning Sqft'][job_app['Existing Zoning Sqft']>1e+5],'\n\n')

# Plot the histogram
fig, axis = plt.subplots(2,3, figsize=(12.1, 5.5))
fig.suptitle(topic, **suptitle_param)

# Sin escala logarítmica y sin rotacion
ax = axis[0,0]
ax.set_title("Basic Histogram", **title_param)
job_app['Existing Zoning Sqft'].plot(kind='hist', rwidth=.7, ax=ax)
ax.set_xlabel('Existing Zoning Sqft')

# Con rotación y sin escala logarítmica
ax = axis[0,1]
ax.set_title("With rotation", **title_param)
job_app['Existing Zoning Sqft'].plot(kind='hist', rwidth=.7, rot=70, ax=ax)
ax.set_xlabel('Existing Zoning Sqft')

# Con rotación y sin escala logarítmica
ax = axis[0,2]
ax.set_title("With rotation", **title_param)
job_app['Existing Zoning Sqft'].plot(kind='hist', rwidth=.7, rot=90, ax=ax)
ax.set_xlabel('Existing Zoning Sqft')

# Con escala logarítmica en x
ax = axis[1,0]
ax.set_title("With log scale on x axis", **title_param)
job_app['Existing Zoning Sqft'].plot(kind='hist', rwidth=.7, logx=True, logy=False, ax=ax)
ax.set_xlabel('Existing Zoning Sqft')

# Con rotación y con escala logarítmica en ambos ejes
ax = axis[1,1]
ax.set_title("With rotation and log scale", **title_param)
job_app['Existing Zoning Sqft'].plot(kind='hist', rwidth=.9, rot=70, logx=True, logy=True, ax=ax)
ax.set_xlabel('Existing Zoning Sqft')

ax = axis[1,2]
ax.set_title("With rotation and auto log scale", **title_param)
job_app['Existing Zoning Sqft'].plot(kind='hist', rwidth=.9, rot=70, log=True, ax=ax)
ax.set_xlabel('Existing Zoning Sqft')

# Display the graph
plt.subplots_adjust(left=None, bottom=.25, right=None, top=.8, wspace=.5, hspace=1.2); #To set the margins 
plt.show()



print("****************************************************")
topic = "9. Visualizing multiple variables with boxplots"; print("** %s\n" % topic)

"""
Visualizing multiple variables with boxplots
Histograms are great ways of visualizing single variables. To visualize multiple variables, 
boxplots are useful, especially when one of the variables is categorical.
In this exercise, your job is to use a boxplot to compare the 'initial_cost' across the 
different values of the 'Borough' column. The pandas .boxplot() method is a quick way to 
do this, in which you have to specify the column and by parameters. Here, you want to 
visualize how 'initial_cost' varies by 'Borough'.
pandas and matplotlib.pyplot have been imported for you as pd and plt, respectively, 
and the DataFrame has been pre-loaded as df.
"""
# Create the boxplot
fig, ax = plt.subplots()
job_app.boxplot(column="Proposed Zoning Sqft", by="Borough", rot=90, ax=ax)
ax.set_ylabel('Proposed Zoning Sqft')
fig.suptitle(topic, **suptitle_param)
ax.set_title("Proposed Zoning Sqft by Borough", **title_param)
plt.subplots_adjust(left=None, bottom=.25, right=None, top=.85, wspace=.5, hspace=1.2); #To set the margins 
plt.show()

print(job_app["Proposed Zoning Sqft"].describe(),'\n\n')
print(job_app["Borough"].describe(),'\n\n')
print(job_app[job_app["Proposed Zoning Sqft"]>1e+6],'\n\n')



print("****************************************************")
topic = "10. Visualizing multiple variables with scatter plots"; print("** %s\n" % topic)

"""
Visualizing multiple variables with scatter plots
Boxplots are great when you have a numeric column that you want to compare across 
different categories. When you want to visualize two numeric columns, scatter plots 
are ideal.
In this exercise, your job is to make a scatter plot with 'initial_cost' on the 
x-axis and the 'total_est_fee' on the y-axis. You can do this by using the DataFrame 
.plot() method with kind='scatter'. You'll notice right away that there are 2 major 
outliers shown in the plots.
Since these outliers dominate the plot, an additional DataFrame, df_subset, has been 
provided, in which some of the extreme values have been removed. After making a scatter 
plot using this, you'll find some interesting patterns here that would not have been seen 
by looking at summary statistics or 1 variable plots.
When you're done, you can cycle between the two plots by clicking the 'Previous Plot' and 
'Next Plot' buttons below the plot.
"""
# Create and display the first scatter plot
fig, ax = plt.subplots()
fig.suptitle(topic, **suptitle_param)
ax.set_title("A Scatter Plot", **title_param)
job_app.plot(kind="scatter", x="Proposed Zoning Sqft", y="Proposed Height", rot=70, ax=ax)
plt.subplots_adjust(left=None, bottom=.25, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
plt.show()

print(job_app["Proposed Zoning Sqft"].describe())
print(job_app["Proposed Zoning Sqft"].describe())

plt.style.use('default')



print("****************************************************")
print("** END                                            **")
print("****************************************************")