# -*- coding: utf-8 -*-
# Data Type helps: https://www.numpy.org/devdocs/user/basics.types.html
"""
Created on Sun May  5 07:54:27 2019

@author: jacqueline.cortez
"""

# Import plotting modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import scipy.stats as stats

# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.0f}'.format 
#pd.reset_option("all")

def show_df_info(df):
    df_info = {}
    #df_info["columns"] = df.columns.values
    df_info["null_values"] = df.isnull().sum()
    df_info["not_null_values"] = df.notnull().sum()
    df_info["type_column"] = df.dtypes
    df_info["unique_values"] = df.nunique(dropna=False)
    #df_info["memory_use"] = df.memory_usage()[1:]
    print("Shape: {}".format(df.shape)) #Regular expressions
    print(pd.DataFrame(df_info))
    print("Index {}: From {} to {}".format(df.index.dtype, df.index.min(), df.index.max()))
    print("Total memory used: {0:,.0f}".format(df.memory_usage().sum()))
    print("")

sns.set(rc={'figure.figsize':(5,3)})

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")

# Read the file into a DataFrame named ri
file = "auto-mpg.data"
auto = pd.read_csv(file, sep="\s+", header=None, quotechar='"', na_values="?", 
                   names=["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration","model_year", "origin", "car_name"], 
                   dtype={"mpg": np.float64, "cylinders": np.int64, "displacement": np.float64, "horsepower": np.float64, "weight": np.float64, "acceleration": np.float64,"model_year": np.int64, "origin": str, "car_name": str})
show_df_info(auto)

print("****************************************************")
print("PREPARING AND CLEANING THE DATA...\n")
# Drop all rows that are missing 'driver_gender'
auto.dropna(subset=["horsepower"], inplace=True)
auto.reset_index(inplace=True)

# Re-encode the values for origin (1, 2, 3) into “North America”, “Europe”, and “Asia” 
auto["origin"] = auto.origin.replace(["1", "2", "3"],["North America", "Europe", "Asia"])

# Fixing data types
auto["origin"] = auto.origin.astype("category")
show_df_info(auto)
print(auto.describe())
print(auto.head())

print("****************************************************")
print("SIMPLE LINEAR REGRESSIONS...\n")
# Plot a linear regression between 'weight' and 'hp'
sns.lmplot(x='weight', y='mpg', data=auto)
plt.title("Simple linear regressions", color="red")
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("PLOTTING RESIDUALS OF A REGRESSIONS...\n")
plt.figure()
# Generate a green residual plot of the regression between 'hp' and 'mpg'
sns.residplot(x='weight', y='mpg', data=auto, color='green')
plt.title("Plotting residuals of a regression", color="red")
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("HIGHER-ORDER REGRESSIONS...\n")
plt.figure()
# Generate a scatter plot of 'weight' and 'mpg' using red circles
plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')
# Plot in blue a linear regression of order 1 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, color="blue", scatter=None, label='order 1')
# Plot in green a linear regression of order 2 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, color="green", scatter=None, label='order 2', order=2)
plt.legend(loc="upper right") # Add a legend and display the plot
plt.title("Higher-order regressions", color="red")
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("GROUPING LINEAR REGRESSIONS BY HUE...\n¿")
# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x="weight", y="horsepower", data=auto, hue="origin", palette="Set1", legend=False)
plt.legend(loc="upper left", title="Origin") # Set the legend
plt.title("Grouping linear regressions by hue", color="red")
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("GROUPING LINEAR REGRESSIONS BY ROW OR COLUMN...\n")
# Plot linear regressions between 'weight' and 'hp' grouped row-wise by 'origin'
sns.lmplot(x="weight", y="horsepower", data=auto, row="origin", hue="origin")
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("CONSTRUCTING STRIP PLOTS...\n")
# Make a strip plot of 'hp' grouped by 'cyl'
plt.subplot(2,1,1)
sns.stripplot(x="cylinders", y="horsepower", data=auto)
# Make the strip plot again using jitter and a smaller point size
plt.subplot(2,1,2)
sns.stripplot(x="cylinders", y="horsepower", data=auto, jitter=True, size=3)
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("CONSTRUCTING SWARM PLOTS...\n")
plt.figure()
# Generate a swarm plot of 'hp' grouped horizontally by 'cyl'  
plt.subplot(2,1,1)
sns.swarmplot(x="cylinders", y="horsepower", data=auto)
# Generate a swarm plot of 'hp' grouped vertically by 'cyl' with a hue of 'origin'
plt.subplot(2,1,2)
sns.swarmplot(x="horsepower", y="cylinders", data=auto, hue="origin", orient="h")
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("CONSTRUCTING VIOLIN PLOTS...\n")
plt.figure()
# Generate a violin plot of 'hp' grouped horizontally by 'cyl'
plt.subplot(2,1,1)
sns.violinplot(x="cylinders", y="horsepower", data=auto)
# Generate the same violin plot again with a color of 'lightgray' and without inner annotations
plt.subplot(2,1,2)
sns.violinplot(x="cylinders", y="horsepower", data=auto, inner=None, color="lightgray")
# Overlay a strip plot on the violin plot
sns.stripplot(x="cylinders", y="horsepower", data=auto, jitter=True, size=1.5)
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("PLOTTING JOINT DISTRIBUTIONS (1)...\n")
# Generate a joint plot of 'hp' and 'mpg'
#g = sns.jointplot(x="horsepower", y="mpg", data=auto, kind='reg')
g = sns.jointplot(x="horsepower", y="mpg", data=auto)
#g = g.annotate(stats.pearsonr)
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("PLOTTING JOINT DISTRIBUTIONS (2)...\n")
# Generate a joint plot of 'hp' and 'mpg' using a hexbin plot
g = sns.jointplot(x="horsepower", y="mpg", data=auto, kind="hex")
#g.annotate(stats.pearsonr)
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("PLOTTING DISTRIBUTIONS PAIRWISE (1)...\n")
# Generate a joint plot of 'hp' and 'mpg' using a hexbin plot
sns.pairplot(auto) 
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("PLOTTING DISTRIBUTIONS PAIRWISE (2)...\n")
# Generate a joint plot of 'hp' and 'mpg' using a hexbin plot
sns.pairplot(auto[["mpg","horsepower","origin"]], kind="reg", hue="origin")
plt.show() # Display the plot
#plt.clf() # Clear the graph space

print("****************************************************")
print("** END                                            **")
print("****************************************************")
#pd.reset_option("all")