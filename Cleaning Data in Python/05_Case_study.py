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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys


print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)
 
# Global params
pd.set_option('display.max_rows', 10000) #Shows all rows
suptitle_param = dict(color='darkblue', fontsize=12)
title_param = {'color': 'darkred', 'fontsize': 14}

plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 8, 'font.size': 8})

# Read the data
tips = pd.read_csv('tips.csv')

cols = ['Life expectancy', '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807', '1808', '1809', '1810', '1811', '1812', '1813', '1814', '1815', '1816', '1817', '1818', '1819', '1820', '1821', '1822', '1823', '1824', '1825', '1826', '1827', '1828', '1829', '1830', '1831', '1832', '1833', '1834', '1835', '1836', '1837', '1838', '1839', '1840', '1841', '1842', '1843', '1844', '1845', '1846', '1847', '1848', '1849', '1850', '1851', '1852', '1853', '1854', '1855', '1856', '1857', '1858', '1859', '1860', '1861', '1862', '1863', '1864', '1865', '1866', '1867', '1868', '1869', '1870', '1871', '1872', '1873', '1874', '1875', '1876', '1877', '1878', '1879', '1880', '1881', '1882', '1883', '1884', '1885', '1886', '1887', '1888', '1889', '1890', '1891', '1892', '1893', '1894', '1895', '1896', '1897', '1898', '1899'] 
g1800s = pd.read_csv('gapminder.csv', usecols=cols)
g1800s = g1800s[np.concatenate((list(g1800s.columns[-1:].values),list(g1800s.columns[:-1].values)))]
cols = ['Life expectancy', '1900', '1901', '1902', '1903', '1904', '1905', '1906', '1907', '1908', '1909', '1910', '1911', '1912', '1913', '1914', '1915', '1916', '1917', '1919', '1919', '1920', '1921', '1922', '1923', '1924', '1925', '1926', '1927', '1928', '1929', '1930', '1931', '1932', '1933', '1934', '1935', '1936', '1937', '1938', '1939', '1940', '1941', '1942', '1943', '1944', '1945', '1946', '1947', '1948', '1949', '1950', '1951', '1952', '1953', '1954', '1955', '1956', '1957', '1958', '1959', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999'] 
g1900s = pd.read_csv('gapminder.csv', usecols=cols)
g1900s = g1900s[np.concatenate((list(g1900s.columns[-1:].values),list(g1900s.columns[:-1].values)))]
cols = ['Life expectancy', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016'] 
g2000s = pd.read_csv('gapminder.csv', usecols=cols)
g2000s = g2000s[np.concatenate((list(g2000s.columns[-1:].values),list(g2000s.columns[:-1].values)))]

print("****************************************************")
topic = "1. Putting it all together"; print("** %s\n" % topic)

print(tips.head(),'\n\n')
print(tips.info(),'\n\n')
print(tips.columns,'\n\n')
print(tips.describe(),'\n\n')
print(tips.sex.value_counts(),'\n\n')


fig, ax = plt.subplots()
fig.suptitle(topic, **suptitle_param)
tips.tip.plot(kind='hist', rwidth=.9, ax=ax)
ax.set_title('EDA - Hist Tip', **title_param)
ax.set_xlabel('Tip')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None); #To set the margins 
plt.show()
      

print("****************************************************")
topic = "2. Exploratory analysis"; print("** %s\n" % topic)
"""
Exploratory analysis
Whenever you obtain a new dataset, your first task should always be to 
do some exploratory analysis to get a better understanding of the data 
and diagnose it for any potential issues.
The Gapminder data for the 19th century has been loaded into a DataFrame 
called g1800s. In the IPython Shell, use pandas methods such as .head(), 
.info(), and .describe(), and DataFrame attributes like .columns and 
.shape to explore it.
Use the information that you acquire from your exploratory analysis to 
choose the true statement from the options provided below.
"""
print(g1800s.info(),'\n\n')
print(g1800s.columns,'\n\n')
print(g1800s.head(),'\n\n')
print(g1800s.describe(),'\n\n')



print("****************************************************")
topic = "3. Visualizing your data"; print("** %s\n" % topic)
"""
Visualizing your data
Since 1800, life expectancy around the globe has been steadily going up. 
You would expect the Gapminder data to confirm this.
The DataFrame g1800s has been pre-loaded. Your job in this exercise is to 
create a scatter plot with life expectancy in '1800' on the x-axis and 
life expectancy in '1899' on the y-axis.
Here, the goal is to visually check the data for insights as well as 
errors. When looking at the plot, pay attention to whether the scatter 
plot takes the form of a diagonal line, and which points fall below or 
above the diagonal line. This will inform how life expectancy in 1899 
changed (or did not change) compared to 1800 for different countries. 
If points fall on a diagonal line, it means that life expectancy remained 
the same!
"""
fig, ax = plt.subplots()
fig.suptitle(topic, **suptitle_param)
ax.set_title('EDA - Hist g1800s', **title_param)

# Create the scatter plot
g1800s.plot(kind="scatter", x="1800", y="1899", ax=ax)

# Specify axis labels
ax.set_xlabel('Life Expectancy by Country in 1800')
ax.set_ylabel('Life Expectancy by Country in 1899')

# Specify axis limits
ax.set_xlim(20, 55)
ax.set_ylim(20, 55)

# Display the plot
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None); #To set the margins 
plt.show()



print("****************************************************")
topic = "4. Thinking about the question at hand"; print("** %s\n" % topic)
"""
Thinking about the question at hand
Since you are given life expectancy level data by country and year, you 
could ask questions about how much the average life expectancy changes 
over each year.
Before continuing, however, it's important to make sure that the following 
assumptions about the data are true:
'Life expectancy' is the first column (index 0) of the DataFrame.
The other columns contain either null or numeric values.
The numeric values are all greater than or equal to 0.
There is only one instance of each country.
You can write a function that you can apply over the entire DataFrame to 
verify some of these assumptions. Note that spending the time to write 
such a script will help you when working with other datasets as well.
"""
def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
    no_na = row_data.dropna()
    if len(no_na)>0:
        numeric = pd.to_numeric(row_data, errors="coerce")
        ge0 = numeric >= 0
        return ge0
    else:
        return False



# Check whether the first column is 'Life expectancy'
try:
    print(g1800s.columns[0],'\n\n')
    assert g1800s.columns[0] == "Life expectancy"
except:
    print("Error msg - columns 0...\n\n")


# Check whether the values in the row are valid
try:
    assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=0).all().all()
except:
    print("Error msg - check nulls...")

g1800s = g1800s.groupby('Life expectancy').sum().reset_index()
err_msg = ""
try:
    assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=0).all().all()
except:
    err_msg="Error msg - check nulls..."; print(err_msg)
print("...Fixed, now its ok!\n\n" if err_msg=="" else "\n\n")

    
# Check that there is only one instance of each country
try:
    print(g1800s['Life expectancy'].value_counts().head(),'\n\n')
    assert g1800s['Life expectancy'].value_counts()[0] == 1
except:
    print('Error msg - value counts...\n\n')



print("****************************************************")
topic = "5. Assembling your data"; print("** %s\n" % topic)

# Concatenate the DataFrames row-wise
gapminder = pd.concat([g1800s,g1900s,g2000s])

# Print the shape of gapminder
print(gapminder.shape, '\n\n')

# Print the head and the tail of gapminder
print(gapminder.head(), '\n\n')
print(gapminder.tail(), '\n\n')

print(gapminder.info(verbose=True, null_counts=True),'\n\n')
print(gapminder.columns,'\n\n')
print(gapminder.describe(),'\n\n')
print(gapminder.dtypes,'\n\n')


print("****************************************************")
topic = "6. Initial impressions of the data"; print("** %s\n" % topic)

# Checking data types
print(tips.dtypes,'\n\n')

tips['tip'] = tips['tip'].astype(str)
print(tips.dtypes,'\n\n')

tips['tip'] = pd.to_numeric(tips.tip, errors='coerce')
print(tips.dtypes,'\n\n')


#Additional calculations and saving your data
tips['grand_total'] = tips.total_bill + tips.tip

def double_tip(tip):
    try:    return tip*2
    except: return 0
tips['double_tip'] = tips.tip.apply(double_tip)

def grand_tot_double_tip(row):
    try:    return row['total_bill'] + row['double_tip']
    except: return 0
tips['grand_total_double_tip'] = tips.apply(grand_tot_double_tip, axis=1)

print(tips.dtypes,'\n\n')
print(tips.head(),'\n\n')

file = 'tips_saved.csv'
tips.to_csv(file)



print("****************************************************")
topic = "7. Reshaping your data"; print("** %s\n" % topic)
"""
Reshaping your data
Now that you have all the data combined into a single DataFrame, the next step 
is to reshape it into a tidy data format.
Currently, the gapminder DataFrame has a separate column for each year. What 
you want instead is a single column that contains the year, and a single column 
that represents the average life expectancy for each year and country. By 
having year in its own column, you can use it as a predictor variable in a 
later analysis.
You can convert the DataFrame into the desired tidy format by melting it.
"""
# Melt gapminder: gapminder_melt
gapminder_melt = pd.melt(frame=gapminder, id_vars="Life expectancy")
print(gapminder_melt.head())

# Rename the columns
gapminder_melt.columns = ["country","year","life_expectancy"]

# Print the head of gapminder_melt
print(gapminder_melt.head(),'\n\n')
print(gapminder_melt.info(),'\n\n')



print("****************************************************")
topic = "8. Checking the data types"; print("** %s\n" % topic)
"""
Checking the data types
Now that your data are in the proper shape, you need to ensure that the columns 
are of the proper data type. That is, you need to ensure that country is of 
type object, year is of type int64, and life_expectancy is of type float64.
The tidy DataFrame has been pre-loaded as gapminder. Explore it in the IPython 
Shell using the .info() method. Notice that the column 'year' is of type object. 
This is incorrect, so you'll need to use the pd.to_numeric() function to convert 
it to a numeric data type.
NumPy and pandas have been pre-imported as np and pd.
"""
# Convert the year column to numeric
try: assert gapminder_melt.year.dtypes == np.int64
except: print('year column will be transformed to int64 type...\n\n')
gapminder_melt['year'] = pd.to_numeric(gapminder_melt.year, errors='coerce')
print(gapminder_melt.info(),'\n\n')

# Test if country is of type object
try: assert gapminder_melt.country.dtypes == np.object
except: print('country column is not object type...\n\n')

# Test if year is of type int64
try: assert gapminder_melt.year.dtypes == np.int64
except: print('year column is not int64 type...\n\n')

# Test if life_expectancy is of type float64
try: assert gapminder_melt.life_expectancy.dtypes == np.float64
except: print('life_expectancy is not float64 type...\n\n')



print("****************************************************")
topic = "9. Looking at country spellings"; print("** %s\n" % topic)
"""
Looking at country spellings
Having tidied your DataFrame and checked the data types, your next task in the 
data cleaning process is to look at the 'country' column to see if there are 
any special or invalid characters you may need to deal with.
It is reasonable to assume that country names will contain:
The set of lower and upper case letters.
Whitespace between words.
Periods for any abbreviations.
To confirm that this is the case, you can leverage the power of regular 
expressions again. For common operations like this, Pandas has a built-in 
string method - str.contains() - which takes a regular expression pattern, and 
applies it to the Series, returning True if there is a match, and False 
otherwise.
Since here you want to find the values that do not match, you have to invert 
the boolean, which can be done using ~. This Boolean series can then be used to 
get the Series of countries that have invalid names.
"""
# Create the series of countries: countries
countries = gapminder_melt["country"]
print("***SAMPLE DATA***")
print(countries.head())
print("Cantidad de paises recopilados:", countries.shape, '\n\n')

# Drop all the duplicates from countries
countries = countries.drop_duplicates()
print("***NO DUPLICATE COUNTRIES***")
print('Sample of no duplicated countries: \n{}'.format(countries.iloc[212:217]))
print('... See row from 212 to 214 are valid, meanwhile 215 are not')
print(countries.shape, '\n\n')


# Find invalid names of countries
print("***INVALID COUNTRY'S NAMES***")
pattern = '^[A-Z][A-Za-z\s\.]+$' # A-Za-z, space and point are valid for the name.
mask = countries.str.contains(pattern) # Create the Boolean vector: mask
mask_inverse = ~mask ## Invert the mask: mask_inverse
invalid_countries = countries.loc[mask_inverse] # Subset countries using mask_inverse: invalid_countries

# Print invalid_countries
print(invalid_countries)
print(invalid_countries.shape, '\n\n')



print("****************************************************")
topic = "10. More data cleaning and processing"; print("** %s\n" % topic)
"""
More data cleaning and processing
It's now time to deal with the missing data. There are several strategies for 
this: You can drop them, fill them in using the mean of the column or row that 
the missing value is in (also known as imputation), or, if you are dealing with 
time series data, use a forward fill or backward fill, in which you replace 
missing values in a column with the most recent known value in the column. 
See pandas Foundations for more on forward fill and backward fill.
In general, it is not the best idea to drop missing values, because in doing so 
you may end up throwing away useful information. In this data, the missing 
values refer to years where no estimate for life expectancy is available for a 
given country. You could fill in, or guess what these life expectancies could 
be by looking at the average life expectancies for other countries in that year, 
for example. Whichever strategy you go with, it is important to carefully 
consider all options and understand how they will affect your data.
In this exercise, you'll practice dropping missing values. Your job is to drop 
all the rows that have NaN in the life_expectancy column. Before doing so, it 
would be valuable to use assert statements to confirm that year and country do 
not have any missing values.
Begin by printing the shape of gapminder in the IPython Shell prior to dropping 
the missing values. Complete the exercise to find out what its shape will be 
after dropping the missing values!
"""
print("Tamaño del DataFrame antes de la depuración:", gapminder_melt.shape, '\n\n')
tamaño_inicial=gapminder_melt.shape[0]

# Assert that country does not contain any missing values
try   : assert pd.notnull(gapminder_melt.country).all()
except: print("Country contain missing values (pd.notnull)...")
try   : assert gapminder_melt.country.notnull().all()
except: print("Country contains missing values (pd.DataFrame.notnull)...")

# Assert that year does not contain any missing values
try   : assert pd.notnull(gapminder_melt.year).all()
except: print("Year contains missing values...")
# Drop the missing values
gapminder_melt.dropna(inplace=True)

# Print the shape of gapminder
print("Tamaño del DataFrame despues de la depuración:", gapminder_melt.shape, '\n\n')
tamaño_final=gapminder_melt.shape[0]

print("Se reduce en un {:,.2%}\n\n".format(tamaño_final/tamaño_inicial))



print("****************************************************")
topic = "11. Wrapping up"; print("** %s\n" % topic)
"""
Wrapping up
Now that you have a clean and tidy dataset, you can do a bit of visualization 
and aggregation. In this exercise, you'll begin by creating a histogram of the 
life_expectancy column. You should not get any values under 0 and you should 
see something reasonable on the higher end of the life_expectancy age range.
Your next task is to investigate how average life expectancy changed over the 
years. To do this, you need to subset the data by each year, get the 
life_expectancy column from each subset, and take an average of the values. 
You can achieve this using the .groupby() method. This .groupby() method is 
covered in greater depth in Manipulating DataFrames with pandas.
Finally, you can save your tidy and summarized DataFrame to a file using the 
.to_csv() method.
matplotlib.pyplot and pandas have been pre-imported as plt and pd. Go for it!
"""
# Group gapminder: gapminder_agg
gapminder_agg = gapminder_melt.groupby('year')['life_expectancy'].mean()
# Print the head of gapminder_agg
print(gapminder_agg.head(), '\n\n')
# Print the tail of gapminder_agg
print(gapminder_agg.tail(), '\n\n')

# Add first subplot
fig, axis = plt.subplots(1, 2, figsize=(10,4))
fig.suptitle(topic, **suptitle_param)

# Create a histogram of life_expectancy
ax=axis[0]
gapminder_melt.life_expectancy.plot(kind='hist', rwidth=.9, ax=ax)
ax.set_title('EDA - Hist life_expectancy', **title_param)
ax.set_xlabel('life_expectancy')


# Create a line plot of life expectancy per year
ax=axis[1]
gapminder_agg.plot(ax=ax)
ax.set_title('Life expectancy over the years', **title_param)
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, wspace=None, hspace=None); #To set the margins 
plt.show()

# Save both DataFrames to csv files
file = "gapminder_saved.csv"; gapminder_melt.to_csv(file); print("'{}' file saved...".format(file))
file = "gapminder_agg_saved.csv"; gapminder_agg.to_csv(file); print("'{}' file saved...".format(file))



print("****************************************************")
topic = "12. Final thoughts"; print("** %s\n" % topic)

plt.style.use('default')



print("****************************************************")
print("** END                                            **")
print("****************************************************")