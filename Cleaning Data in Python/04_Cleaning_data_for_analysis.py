# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:52:04 2020

@author: jacesca@gmail.com
Chapter4 - Cleaning data for analysis:
    Dive into some of the grittier aspects of data cleaning. 
    Learn about string manipulation and pattern matching to deal with unstructured data, 
    and then explore techniques to deal with missing or duplicate data. You'll also learn 
    the valuable skill of programmatically checking your data for consistency, which will 
    give you confidence that your code is running correctly and that the results of your 
    analysis are reliable.
Source: https://learn.datacamp.com/courses/cleaning-data-in-python

"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import numpy as np
import pandas as pd
import re
import sys


print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)
 
pd.set_option('display.max_rows', 100) #Shows all rows

# Read the data
job_app = pd.read_csv('dob_job_application_filings_subset.csv', low_memory=False)
tips = pd.read_csv('tips.csv')
tips_nan = pd.read_csv('tips_nan.csv')
billboard_tidy = pd.read_csv('billboard_tidy.csv')
billboard = pd.read_csv('billboard.csv', low_memory=False)
airquality = pd.read_csv('airquality.csv')

mini_job_app = job_app[['Borough', 'Initial Cost', 'Total Est. Fee']].copy()


print("****************************************************")
topic = "1. Data types"; print("** %s\n" % topic)

print(job_app.dtypes, '\n\n')
print(job_app.columns, '\n\n')

# Turning to string
print(job_app["Owner'sPhone "].head(), '\n\n')
job_app["Owner'sPhone "] = job_app["Owner'sPhone "].astype(str)
print(job_app["Owner'sPhone "].head(), '\n\n')

# Turning to numeric
print(job_app['Initial Cost'].head(), '\n\n')
job_app['Initial Cost'] = job_app['Initial Cost'].str[1:]
print(job_app['Initial Cost'].head(), '\n\n')
job_app['Initial Cost'] = pd.to_numeric(job_app['Initial Cost'], errors='coerce')
print(job_app['Initial Cost'].head(), '\n\n')

job_app['Total Est. Fee'] = job_app['Total Est. Fee'].str[1:]
job_app['Total Est. Fee'] = pd.to_numeric(job_app['Total Est. Fee'], errors='coerce')

print(job_app.dtypes, '\n\n')



print("****************************************************")
topic = "2. Converting data types"; print("** %s\n" % topic)
"""
Converting data types
In this exercise, you'll see how ensuring all categorical 
variables in a DataFrame are of type category reduces memory 
usage.
The tips dataset has been loaded into a DataFrame called tips. 
This data contains information about how much a customer tipped, 
whether the customer was male or female, a smoker or not, etc.
Look at the output of tips.info() in the IPython Shell. You'll 
note that two columns that should be categorical - sex and smoker 
- are instead of type object, which is pandas' way of storing 
arbitrary strings. Your job is to convert these two columns to 
type category and note the reduced memory usage.
"""
print(tips.dtypes, '\n\n')
print(tips.columns, '\n\n')

# Convert the sex column to type 'category'
print(tips.sex.unique(), '\n\n')
print(tips.sex.head(), '\n\n')
print(tips.sex.value_counts(dropna=False),'\n\n')
tips.sex = tips.sex.astype("category") #memory usage: 13.8+ KB
print(tips.sex.head(), '\n\n')

# Convert the smoker column to type 'category'
print(tips.smoker.unique(), '\n\n')
print(tips.smoker.head(), '\n\n')
print(tips.smoker.value_counts(dropna=False),'\n\n')
tips.smoker = tips.smoker.astype("category")

# Print the info of tips
print(tips.info(),'\n\n')


print("****************************************************")
topic = "3. Working with numeric data"; print("** %s\n" % topic)
"""
Working with numeric data
If you expect the data type of a column to be numeric (int or 
float), but instead it is of type object, this typically means 
that there is a non numeric value in the column, which also 
signifies bad data.
You can use the pd.to_numeric() function to convert a column 
into a numeric data type. If the function raises an error, you 
can be sure that there is a bad value within the column. You can 
either use the techniques you learned in Chapter 1 to do some 
exploratory data analysis and find the bad value, or you can 
choose to ignore or coerce the value into a missing value, NaN.
A modified version of the tips dataset has been pre-loaded into 
a DataFrame called tips. For instructional purposes, it has been 
pre-processed to introduce some 'bad' data for you to clean. Use 
the .info() method to explore this. You'll note that the 
total_bill and tip columns, which should be numeric, are instead 
of type object. Your job is to fix this.
"""
# Convert 'total_bill' and 'tip' to an object dtype
tips.total_bill = tips.total_bill.astype(str)
tips.tip = tips.tip.astype(str)
print(tips.info(),'\n\n')

# Convert 'total_bill' and 'tip' to a numeric dtype
tips.total_bill = pd.to_numeric(tips.total_bill, errors="coerce")
tips.tip = pd.to_numeric(tips.tip, errors="coerce")

# Print the info of tips
print(tips.info(),'\n\n')


print("****************************************************")
topic = "4. Using regular expressions to clean strings"; print("** %s\n" % topic)

pattern = re.compile('\$\d*\.\d{2}')
result = pattern.match('$17.89')
print(bool(result),'\n\n')


print("****************************************************")
topic = "5. String parsing with regular expressions"; print("** %s\n" % topic)
"""
String parsing with regular expressions
In the video, Dan introduced you to the basics of regular 
expressions, which are powerful ways of defining patterns to 
match strings. This exercise will get you started with writing 
them.
When working with data, it is sometimes necessary to write a 
regular expression to look for properly entered values. Phone 
numbers in a dataset is a common field that needs to be checked 
for validity. Your job in this exercise is to define a regular 
expression to match US phone numbers that fit the pattern of 
xxx-xxx-xxxx.
The regular expression module in python is re. When performing 
pattern matching on data, since the pattern will be used for a 
match across multiple rows, it's better to compile the pattern 
first using re.compile(), and then use the compiled pattern to 
match values.
"""
# Compile the pattern: prog
prog = re.compile('^\d{3}-\d{3}-\d{4}$')

# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result),'\n\n')

# See if the pattern matches
result2 = prog.match('1123-456-7890')
print(bool(result2),'\n\n')


print("****************************************************")
topic = "6. Extracting numerical values from strings"; print("** %s\n" % topic)
"""
Extracting numerical values from strings
Extracting numbers from strings is a common task, particularly 
when working with unstructured data or log files.
Say you have the following string: 'the recipe calls for 6 
strawberries and 2 bananas'.
It would be useful to extract the 6 and the 2 from this string 
to be saved for later use when comparing strawberry to banana 
ratios.
When using a regular expression to extract multiple numbers (or 
multiple pattern matches, to be exact), you can use the 
re.findall() function. Dan did not discuss this in the video, 
but it is straightforward to use: You pass in a pattern and a 
string to re.findall(), and it will return a list of the matches.
"""
# Find the numeric values: matches
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches,'\n\n')


print("****************************************************")
topic = "7. Pattern matching"; print("** %s\n" % topic)
"""
Pattern matching
In this exercise, you'll continue practicing your regular 
expression skills. For each provided string, your job is to 
write the appropriate pattern to match it.
"""
# Write the first pattern
pattern1 = bool(re.match(pattern='^\d{3}-\d{3}-\d{4}$', string='123-456-7890'))
print(pattern1,'\n\n')

# Write the second pattern
pattern2 = bool(re.match(pattern='^\$\d*\.\d{2}$', string='$123.45'))
print(pattern2,'\n\n')

# Write the third pattern
pattern3 = bool(re.match(pattern='^[A-Z]\w*', string='Australia'))
print(pattern3,'\n\n')
pattern3 = bool(re.match(pattern='^[A-Z]\w*', string='-Australia'))
print(pattern3,'\n\n')



print("****************************************************")
topic = "8. Using functions to clean data"; print("** %s\n" % topic)

# Applying functions
tips_by_sex = tips.groupby('sex')[['total_bill', 'tip']].sum()
print(tips_by_sex.dtypes, '\n\n')
print(tips_by_sex.head(), '\n\n')

print(tips_by_sex.apply(np.mean, axis=0), '\n\n')
print(tips_by_sex.apply(np.mean, axis=1), '\n\n')


# Write the regular expression
pattern = re.compile('^\$\d*\.\d{2}$')

# Write the function
def diff_money(row, pattern):
    icost = row['Initial Cost']
    tef = row['Total Est. Fee']
    if bool(pattern.match(icost)) and bool(pattern.match(tef)):
        icost = icost.replace("$", "")
        tef = tef.replace("$", "")
        icost = float(icost)
        tef = float(tef)
        return icost - tef
    else:
        return(float('nan'))

print(mini_job_app.dtypes, '\n\n')
print(mini_job_app.head(), '\n\n')

mini_job_app['diff'] = mini_job_app.apply(diff_money, axis=1, pattern=pattern)

print(mini_job_app.dtypes, '\n\n')
print(mini_job_app.head(), '\n\n')



print("****************************************************")
topic = "9. Custom functions to clean data"; print("** %s\n" % topic)
"""
Custom functions to clean data
You'll now practice writing functions to clean data.
The tips dataset has been pre-loaded into a DataFrame called tips. 
It has a 'sex' column that contains the values 'Male' or 'Female'. 
Your job is to write a function that will recode 'Female' to 0, 
'Male' to 1, and return np.nan for all entries of 'sex' that are 
neither 'Female' nor 'Male'.
Recoding variables like this is a common data cleaning task. 
Functions provide a mechanism for you to abstract away complex 
bits of code as well as reuse code. This makes your code more 
readable and less error prone.
As Dan showed you in the videos, you can use the .apply() method 
to apply a function across entire rows or columns of DataFrames. 
However, note that each column of a DataFrame is a pandas Series. 
Functions can also be applied across Series. Here, you will apply 
your function over the 'sex' column.
"""
# Define recode_gender()
def recode_gender(gender):
    if gender == "Female": return 0 # Return 0 if gender is 'Female'
    elif gender == "Male": return 1 # Return 1 if gender is 'Male'    
    else: return np.nan             # Return np.nan    

# Apply the function to the sex column
print(tips.sex.value_counts(dropna=False), '\n\n')

#############################################################
## Warning, using the code:
##    tips.recode = tips.sex.apply(recode_gender)    
## Msg = "UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access"
#############################################################
tips['recode'] = tips.sex.apply(recode_gender)

# Print the first five rows of tips
print(tips.head(), '\n\n')

print(tips.recode.value_counts(dropna=False), '\n\n')
print(tips[pd.Series.isnull(tips["recode"])].head(), '\n\n')




# Define recode_gender()
def recode_gender2(row):
    if row.sex == "Female": return 0 # Return 0 if gender is 'Female'
    elif row.sex == "Male": return 1 # Return 1 if gender is 'Male'    
    else: return float('nan')        # Return np.nan    
tips['recode2'] = tips.apply(recode_gender2, axis=1)
print(tips.head(), '\n\n')
print(tips.recode2.value_counts(dropna=False), '\n\n')



print("****************************************************")
topic = "10. Lambda functions"; print("** %s\n" % topic)
"""
Lambda functions
You'll now be introduced to a powerful Python feature that will help you 
clean your data more effectively: lambda functions. Instead of using the 
def syntax that you used in the previous exercise, lambda functions let 
you make simple, one-line functions.
For example, here's a function that squares a variable used in an 
.apply() method:

    def my_square(x):
        return x ** 2
    
    df.apply(my_square)

The equivalent code using a lambda function is:
    df.apply(lambda x: x ** 2)

The lambda function takes one parameter - the variable x. The function 
itself just squares x and returns the result, which is whatever the one 
line of code evaluates to. In this way, lambda functions can make your 
code concise and Pythonic.
The tips dataset has been pre-loaded into a DataFrame called tips. Your 
job is to clean its 'total_dollar' column by removing the dollar sign. 
You'll do this using two different methods: With the .replace() method, 
and with regular expressions. The regular expression module re has been 
pre-imported.
"""

# Write the lambda function using replace
mini_job_app['Total_replace'] = mini_job_app['Total Est. Fee'].apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
mini_job_app['Total_re'] = mini_job_app['Total Est. Fee'].apply(lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(mini_job_app.head(), '\n\n')




tips['total_dollar'] = tips['total_bill'].apply(lambda x: '$' + str(x))
# Print the head of tips
print(tips.head(), '\n\n')

# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head(), '\n\n')



print("****************************************************")
topic = "11. Duplicate and missing data"; print("** %s\n" % topic)

print(tips_nan.info(), '\n\n')

# First method - dropped
tips_dropped = tips_nan.dropna()
print(tips_dropped.info(), '\n\n')

# Second method - default values
tips_nan['sex'] = tips_nan['sex'].fillna('missing')
tips_nan[['total_bill', 'size']] = tips_nan[['total_bill','size']].fillna(0)
print(tips_nan.info(), '\n\n')

# Third method - statistic values
mean_value = tips_nan['tip'].mean()
print(mean_value, '\n\n')
tips_nan['tip'] = tips_nan['tip'].fillna(mean_value)
print(tips_nan.info(), '\n\n')
print(tips_nan.tail(), '\n\n')


print("****************************************************")
topic = "12. Dropping duplicate data"; print("** %s\n" % topic)
"""
Dropping duplicate data
Duplicate data causes a variety of problems. From the point of view of 
performance, they use up unnecessary amounts of memory and cause unneeded 
calculations to be performed when processing data. In addition, they can 
also bias any analysis results.
A dataset consisting of the performance of songs on the Billboard charts 
has been pre-loaded into a DataFrame called billboard. Check out its 
columns in the IPython Shell. Your job in this exercise is to subset this 
DataFrame and then drop all duplicate rows.
"""
# Rename column
tracks = billboard_tidy[["year","artist.inverted"]].rename(columns={'artist.inverted':'artist'})
print(tracks.info(),'\n\n')


# Create the new DataFrame: tracks
tracks = billboard[["Year","Artist","Track","Time"]]

# Print info of tracks
print(tracks.info(),'\n\n')

# Drop the duplicates: tracks_no_duplicates
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks
print(tracks_no_duplicates.info(),'\n\n')


print("****************************************************")
topic = "13. Filling missing data"; print("** %s\n" % topic)
"""
Filling missing data
Here, you'll return to the airquality dataset from Chapter 2. It has been 
pre-loaded into the DataFrame airquality, and it has missing values for 
you to practice filling in. Explore airquality in the IPython Shell to 
checkout which columns have missing values.
It's rare to have a (real-world) dataset without any missing values, and 
it's important to deal with them because certain calculations cannot 
handle missing values while some calculations will, by default, skip over 
any missing values.
Also, understanding how much missing data you have, and thinking about 
where it comes from is crucial to making unbiased interpretations of data.
"""
# Print the info of airquality
print(airquality.info(),'\n\n')

# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality.Ozone.mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality.Ozone.fillna(oz_mean)

# Print the info of airquality
print(airquality.info(),'\n\n')



print("****************************************************")
topic = "14. Testing with asserts"; print("** %s\n" % topic)

try:
    assert 1 == 1
except:
    print("Error message: 1==1")
    
    

try:
    assert 1 == 2
except OSError as err:
    print("OS error: {0}".format(err))
except ValueError:
    print("Could not convert data to an integer.")
except ZeroDivisionError as err:
    print('Handling run-time error:', err)
except:
    print("Error message: 1==2")
    print(sys.exc_info()[0],'\n\n')



try:
    assert airquality['Solar.R'].notnull().all()
except:
    print("There are null values in column 'Solar.R' from dataframe 'airquality'.")
    print(sys.exc_info()[0], '\n\n')
    
try:
    assert airquality['Ozone'].notnull().all()
except:
    print("There are null values in column 'Ozone' from dataframe 'airquality'.\n\n")

try:
    raise NameError('HiThere')
except NameError:
    print('An exception flew by!\n\n')
    


def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        print("division by zero!")
    except:
        print("variable doesn't match!")
    else:
        print("result is", result)
    finally:
        print("...executing finally clause")

divide(2, 1)
divide(2, 0)
divide("2", "1")



print("****************************************************")
topic = "15. Testing your data with asserts"; print("** %s\n" % topic)
"""
Testing your data with asserts
Here, you'll practice writing assert statements using the Ebola dataset 
from previous chapters to programmatically check for missing values and 
to confirm that all values are positive. The dataset has been pre-loaded 
into a DataFrame called ebola.
In the video, you saw Dan use the .all() method together with the 
.notnull() DataFrame method to check for missing values in a column. 
The .all() method returns True if all values are True. When used on a 
DataFrame, it returns a Series of Booleans - one for each column in the 
DataFrame. So if you are using it on a DataFrame, like in this exercise, 
you need to chain another .all() method so that you return only one True 
or False value. When using these within an assert statement, nothing will 
be returned if the assert statement is true: This is how you can confirm 
that the data you are checking are valid.
Note: You can use pd.notnull(df) as an alternative to df.notnull().
"""
example_airquality = airquality[['Ozone', 'Wind', 'Temp', 'Month', 'Day']]
print(example_airquality.info(),'\n\n')

# Assert that there are no missing values
try:
    assert pd.notnull(example_airquality).all().all()
except:
    print("There are null values.\n\n")

try:
    assert example_airquality.notnull().all().all()
except:
    print("There are null values.\n\n")



# Assert that all values are >= 0
try:
    assert (example_airquality >= 0).all().all()
except:
    print("Not all values are greater than 0.\n\n")

print("****************************************************")
print("** END                                            **")
print("****************************************************")