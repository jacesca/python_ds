# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:52:04 2020

@author: jacesca@gmail.com
Chapter3 - Combining data for analysis:
    The ability to transform and combine your data is a crucial skill in 
    data science, because your data may not always come in one monolithic 
    file or table for you to load. A large dataset may be broken into 
    separate datasets to facilitate easier storage and sharing. But it's 
    important to be able to run your analysis on a single dataset. You'll 
    need to learn how to combine datasets or clean each dataset separately 
    so you can combine them later for analysis.
Source: https://learn.datacamp.com/courses/cleaning-data-in-python

"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import glob
import numpy as np
import pandas as pd


print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

# Readin data
uber1 = pd.read_csv('uber-raw-data-apr14.csv')
uber2 = pd.read_csv('uber-raw-data-may14.csv')
uber3 = pd.read_csv('uber-raw-data-jun14.csv')
ebola = pd.read_csv('ebola.csv')
us_census = pd.read_csv('US_census.csv')
us_state = pd.read_csv('US_state.csv', sep='|')



print("****************************************************")
topic = "1. Concatenating data"; print("** %s\n" % topic)

print(uber1.head(),'\n\n')
print(uber2.head(),'\n\n')

concatenated = pd.concat([uber1, uber2])
print(concatenated.head(), '\n\n')

print(concatenated.loc[0, :], '\n\n')

concatenated = pd.concat([uber1, uber2], ignore_index=True)
print(concatenated.head(), '\n\n')
print(concatenated.loc[0, :], '\n\n')

print("****************************************************")
topic = "2. Combining rows of data"; print("** %s\n" % topic)
"""
Combining rows of data
The dataset you'll be working with here relates to NYC Uber data. 
The original dataset has all the originating Uber pickup locations 
by time and latitude and longitude. For didactic purposes, you'll 
be working with a very small portion of the actual data.
Three DataFrames have been pre-loaded: uber1, which contains data for 
April 2014, uber2, which contains data for May 2014, and uber3, which 
contains data for June 2014. Your job in this exercise is to concatenate 
these DataFrames together such that the resulting DataFrame has the data 
for all three months.
Begin by exploring the structure of these three DataFrames in the IPython 
Shell using methods such as .head().
"""
# Concatenate uber1, uber2, and uber3: row_concat
row_concat = pd.concat([uber1,uber2,uber3])

# Print the shape of row_concat
print(uber1.shape)
print(uber2.shape)
print(uber3.shape)
print(row_concat.shape, '\n\n')

# Print the head of row_concat
print(row_concat.head(), '\n\n')

# Print the tail of row_concat
print(row_concat.tail(), '\n\n')

# Print row with index=0
print(row_concat.loc[0,:], '\n\n')



print("****************************************************")
topic = "3. Combining columns of data"; print("** %s\n" % topic)
"""
Combining columns of data
Think of column-wise concatenation of data as stitching data together 
from the sides instead of the top and bottom. To perform this action, 
you use the same pd.concat() function, but this time with the keyword 
argument axis=1. The default, axis=0, is for a row-wise concatenation.
You'll return to the Ebola dataset you worked with briefly in the last 
chapter. It has been pre-loaded into a DataFrame called ebola_melt. In 
this DataFrame, the status and country of a patient is contained in a 
single column. This column has been parsed into a new DataFrame, 
status_country, where there are separate columns for status and country.
Explore the ebola_melt and status_country DataFrames in the IPython Shell. 
Your job is to concatenate them column-wise in order to obtain a final, 
clean DataFrame.
"""
# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=["Date", "Day"], var_name="type_country", value_name="counts")
ebola_melt['str_split'] = ebola_melt.type_country.str.split("_")
status_country = pd.DataFrame({'status' : ebola_melt["str_split"].str.get(0),
                               'country': ebola_melt["str_split"].str.get(1)})
ebola_melt.drop(columns = ['str_split'], inplace=True)
print(ebola_melt.head(), '\n\n')
print(status_country.head(), '\n\n')

# Concatenate ebola_melt and status_country column-wise: ebola_tidy
ebola_tidy = pd.concat([ebola_melt, status_country],axis=1)

# Print the shape of ebola_tidy
print(ebola_melt.shape)
print(status_country.shape)
print(ebola_tidy.shape, '\n\n')

# Print the head of ebola_tidy
print(ebola_tidy.head(), '\n\n')
print(ebola_tidy.columns, '\n\n')


print("****************************************************")
topic = "4. Finding and concatenating data"; print("** %s\n" % topic)

csv_files = glob.glob('uber-raw-data-*.csv')
print(csv_files, '\n\n')


list_data = []
for filename in csv_files:
    data = pd.read_csv(filename)
    print(data.shape)
    list_data.append(data)

concatenated = pd.concat(list_data)
print(concatenated.shape, '\n\n')
print(concatenated.head(), '\n\n')
print(concatenated.loc[0], '\n\n')


print("****************************************************")
topic = "5. Finding files that match a pattern"; print("** %s\n" % topic)
"""
Finding files that match a pattern
You're now going to practice using the glob module to find all csv files 
in the workspace. In the next exercise, you'll programmatically load them 
into DataFrames.
As Dan showed you in the video, the glob module has a function called glob 
that takes a pattern and returns a list of the files in the working 
directory that match that pattern.
For example, if you know the pattern is part_ single digit number .csv, 
you can write the pattern as 'part_?.csv' (which would match part_1.csv, 
part_2.csv, part_3.csv, etc.)
Similarly, you can find all .csv files with '*.csv', or all parts with 
'part_*'. The ? wildcard represents any 1 character, and the * wildcard 
represents any number of characters.
"""
# Write the pattern: pattern
pattern = 'uber-raw-data-*.csv'

# Save all file matches: csv_files
csv_files = glob.glob(pattern)

# Print the file names
print(csv_files, '\n\n')

# Load the second file into a DataFrame: csv2
csv2 = pd.read_csv(csv_files[1])

# Print the head of csv2
print(csv2.head(), '\n\n')



print("****************************************************")
topic = "6. Iterating and concatenating all matches"; print("** %s\n" % topic)
"""
Iterating and concatenating all matches
Now that you have a list of filenames to load, you can load all the files 
into a list of DataFrames that can then be concatenated.
You'll start with an empty list called frames. Your job is to use a for 
loop to:
iterate through each of the filenames
read each filename into a DataFrame, and then
append it to the frames list.
You can then concatenate this list of DataFrames using pd.concat(). Go for 
it!
"""
# Create an empty list: frames
frames = []

#  Iterate over csv_files
for csv in csv_files:
    #  Read csv into a DataFrame: df
    df = pd.read_csv(csv)
    # Append df to frames
    frames.append(df)

# Concatenate frames into a single DataFrame: uber
uber = pd.concat(frames, ignore_index=True)

# Print the shape of uber
print(uber.shape, '\n\n')

# Print the head of uber
print(uber.head(), '\n\n')
print(uber.tail(), '\n\n')
print(uber.loc[0,:], '\n\n')


print("****************************************************")
topic = "7. Merge data"; print("** %s\n" % topic)

print(us_census.columns,'\n\n')
print(us_state.columns,'\n\n')

df_merged = pd.merge(left=us_census, right=us_state, 
                     on=None, left_on='STATE', right_on='STATE')
print(df_merged.head(), '\n\n')

df_merged = pd.merge(left=us_census, right=us_state, on='STATE')
print(df_merged.head(), '\n\n')



print("****************************************************")
topic = "8. 1-to-1 data merge"; print("** %s\n" % topic)
"""
1-to-1 data merge
Merging data allows you to combine disparate datasets into a 
single dataset to do more complex analysis.
Here, you'll be using survey data that contains readings that 
William Dyer, Frank Pabodie, and Valentina Roerich took in the 
late 1920s and 1930s while they were on an expedition towards 
Antarctica. The dataset was taken from a sqlite database from 
the Software Carpentry SQL lesson.
Two DataFrames have been pre-loaded: site and visited. Explore 
them in the IPython Shell and take note of their structure and 
column names. Your task is to perform a 1-to-1 merge of these 
two DataFrames using the 'name' column of site and the 'site' 
column of visited.
"""
site = pd.DataFrame({'name': ['DR-1', 'DR-3', 'MSK-4'],
                     'lat' : [-49.85, -47.15, -48.87],
                     'long': [-49.85, -47.15, -48.87]})

visited = pd.DataFrame({'ident': [-49.85, -47.15, -48.87],
                        'site' : ['DR-1', 'DR-3', 'MSK-4'],
                        'dated': ['1927-02-08', '1939-01-07', '1932-01-14']})
print(site, '\n\n')
print(visited, '\n\n')

# Merge the DataFrames: o2o
o2o = pd.merge(left=site, right=visited, left_on="name", right_on="site")

# Print o2o
print(o2o, '\n\n')
print(o2o.shape, '\n\n')
print(o2o.info(), '\n\n')


print("****************************************************")
topic = "9. Many-to-1 data merge"; print("** %s\n" % topic)
"""
Many-to-1 data merge
In a many-to-one (or one-to-many) merge, one of the values will 
be duplicated and recycled in the output. That is, one of the 
keys in the merge is not unique.
Here, the two DataFrames site and visited have been pre-loaded 
once again. Note that this time, visited has multiple entries 
for the site column. Confirm this by exploring it in the 
IPython Shell.
The .merge() method call is the same as the 1-to-1 merge from 
the previous exercise, but the data and output will be 
different.
"""
site = pd.DataFrame({'name': ['DR-1', 'DR-3', 'MSK-4'],
                     'lat' : [-49.85, -47.15, -48.87],
                     'long': [-128.57, -126.72, -123.4 ]})

visited = pd.DataFrame({'ident': [619, 622, 734, 735, 751, 752, 837, 844],
                        'site' : ['DR-1', 'DR-1', 'DR-3', 'DR-3', 'DR-3', 'DR-3', 'MSK-4', 'DR-1'],
                        'dated': ['1927-02-08', '1927-02-10', '1939-01-07', '1930-01-12', '1930-02-26', np.nan, '1932-01-14', '1932-03-22']})
print(site, '\n\n')
print(visited, '\n\n')

# Merge the DataFrames: m2o
m2o = pd.merge(left=site, right=visited, left_on="name", right_on="site")

# Print m2o
print(m2o, '\n\n')



print("****************************************************")
topic = "10. Many-to-many data merge"; print("** %s\n" % topic)
"""
Many-to-many data merge
The final merging scenario occurs when both DataFrames do not 
have unique keys for a merge. What happens here is that for 
each duplicated key, every pairwise combination will be created.
Two example DataFrames that share common key values have been 
pre-loaded: df1 and df2. Another DataFrame df3, which is the 
result of df1 merged with df2, has been pre-loaded. All three 
DataFrames have been printed - look at the output and notice 
how pairwise combinations have been created. This example is to 
help you develop your intuition for many-to-many merges.
Here, you'll work with the site and visited DataFrames from 
before, and a new survey DataFrame. Your task is to merge site 
and visited as you did in the earlier exercises. You will then 
merge this merged DataFrame with survey.
Begin by exploring the site, visited, and survey DataFrames in 
the IPython Shell.
"""
survey = pd.DataFrame({'taken'  : [619, 619, 622, 622, 734, 734, 734, 735, 735, 735, 751, 751, 751, 752, 752, 752, 752, 837, 837, 837, 844],
                       'person' : ['dyer', 'dyer', 'dyer', 'dyer', 'pb', 'lake', 'pb', 'pb', np.nan, np.nan, 'pb', 'pb', 'lake', 'lake', 'lake', 'lake', 'roe', 'lake', 'lake', 'roe', 'roe'],
                       'quant'  : ['rad', 'sal', 'rad', 'sal', 'rad', 'sal', 'temp', 'rad', 'sal', 'temp', 'rad', 'temp', 'sal', 'rad', 'sal', 'temp', 'sal', 'rad', 'sal', 'sal', 'rad'],
                       'reading': [  9.82,   0.13,   7.8 ,   0.09,   8.41,   0.05, -21.5 ,   7.22,   0.06, -26.  ,   4.35, -18.5 ,   0.1 ,   2.19,   0.09, -16.  ,   41.6 ,   1.46,   0.21,  22.5 ,  11.25]})

# Merge site and visited: m2m
m2m = pd.merge(left=site, right=visited, left_on="name", right_on="site")
print(m2m.shape, '\n\n')
print(m2m, '\n\n')

# Merge m2m and survey: m2m
m2m = pd.merge(left=m2m, right=survey, left_on="ident", right_on="taken")
print(m2m.shape, '\n\n')

# Print the first 20 lines of m2m
print(m2m.head(20), '\n\n')



print("****************************************************")
print("** END                                            **")
print("****************************************************")