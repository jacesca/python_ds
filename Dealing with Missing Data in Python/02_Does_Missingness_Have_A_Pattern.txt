# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:29:35 2020

@author: jacesca@gmail.com
Chapter 2: Does Missingness Have A Pattern?
    Analyzing the type of missingness in your dataset is a very important step 
    towards treating missing values. In this chapter, you'll learn in detail how 
    to establish patterns in your missing and non-missing data, and how to 
    appropriately treat the missingness using simple techniques such as listwise 
    deletion.
Source: https://learn.datacamp.com/courses/data-manipulation-with-pandas
Help:
    https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py
"""
###############################################################################
## Importing libraries
###############################################################################
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns


###############################################################################
## Preparing the environment
###############################################################################
# Global variables
SEED = 42 

# Global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 8, 'font.size': 8})
suptitle_param   = dict(color='darkblue', fontsize=10)
title_param      = {'color': 'darkred', 'fontsize': 12}

# Reading the data
diabetes = pd.read_csv('pima-indians-diabetes data.csv')

# Functions
def fill_dummy_values(df, scaling_factor=0.075, seed=SEED):
    # Set the seed to replicate the experiment
    np.random.seed(seed)
    # Create copy of dataframe
    df_dummy = df.copy(deep=True)
    
    # Iterate over each column
    for col_name in df_dummy:
        # Get column, column missing values and range
        col = df_dummy[col_name]
        col_null = col.isnull()
        num_nulls = col_null.sum()
        col_range = col.max() - col.min()
        # Shift and scale dummy values
        dummy_values = (np.random.rand(num_nulls) - 2)
        dummy_values = dummy_values * scaling_factor * col_range + col.min()
        # Return dummy values
        df_dummy.loc[col_null, col_name] = dummy_values
    return df_dummy

###############################################################################
## Main part of the code
###############################################################################
def Is_the_data_missing_at_random():
    print("****************************************************")
    topic = "1. Is the data missing at random?"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    print(diabetes.head())
    
    print('---------------------------------------------COUNTING MISSING VALUES')
    counting_missing = diabetes.isna().sum().sort_values(ascending=False)
    columns_with_missing = counting_missing[counting_missing>0].index.tolist()
    column_with_max_missing = counting_missing[counting_missing==counting_missing.max()].index.tolist()
    print(counting_missing)
    
    print('---------------------------------------------FINDING MISSING VALUES')
    #fig, ax = plt.subplots()
    ax = msno.matrix(diabetes, sparkline=True, fontsize=8, figsize=(10, 5.9)) #ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset - Nullity Matrix', **title_param)
    plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    plt.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------FINDING PATTERNS IN THE MISSING VALUES')
    print('---------------------------------------------SORT BY SERUM_INSULIN')
    #diabetes_sort = diabetes.sort_values(['Serum_Insulin'])
    diabetes_sort = diabetes.sort_values(column_with_max_missing)
    #fig, ax = plt.subplots()
    ax = msno.matrix(diabetes_sort, sparkline=True, fontsize=8, figsize=(10, 5.9)) #ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset - Sort Nullity Matrix by Serum_Insulin', **title_param)
    plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    plt.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------FINDING PATTERNS IN THE MISSING VALUES')
    print('---------------------------------------------SORT BY ALL MISSING COLUMNS')
    #diabetes_sort = diabetes.sort_values(['Serum_Insulin', 'Skin_Fold', 'Diastolic_BP', 'BMI', 'Glucose'])
    diabetes_sort = diabetes.sort_values(columns_with_missing)
    #fig, ax = plt.subplots()
    ax = msno.matrix(diabetes_sort, sparkline=True, fontsize=8, figsize=(10, 5.9)) #ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset - Sort Nullity Matrix by all columns with missing values', **title_param)
    plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    plt.suptitle(topic, **suptitle_param)
    plt.show()
    
    
        
def Guess_the_missingness_type():
    print("****************************************************")
    topic = "2. Guess the missingness type"; print("** %s" % topic)
    print("****************************************************")
    
    ax = msno.matrix(diabetes, sparkline=True, fontsize=8, figsize=(10, 5.9)) #ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset - Nullity Matrix', **title_param)
    plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    plt.suptitle(topic, **suptitle_param)
    plt.show()
    
    
        
def Deduce_MNAR():
    print("****************************************************")
    topic = "3. Deduce MNAR"; print("** %s" % topic)
    print("****************************************************")
    
    # Sort diabetes dataframe on 'Serum Insulin'
    sorted_values = diabetes.sort_values('Serum_Insulin')
    # Visualize the missingness summary of sorted
    #fig, ax = plt.subplots()
    ax = msno.matrix(sorted_values, sparkline=True, fontsize=8, figsize=(10, 5.9)) #ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset - Sort Nullity Matrix by Serum_Insulin', **title_param)
    plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    plt.suptitle(topic, **suptitle_param)
    plt.show()
    
    
        
def Finding_patterns_in_missing_data():
    print("****************************************************")
    topic = "4. Finding patterns in missing data"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------HEATMAP')
    fig, ax = plt.subplots(figsize=(10, 5.9))
    msno.heatmap(diabetes, fontsize=8, ax=ax)
    ax.set_title('Pima Diabetes Dataset [Heatmap]', **title_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------DENDROGRAM')
    fig, ax = plt.subplots(figsize=(10, 5.9))
    msno.dendrogram(diabetes, fontsize=8, ax=ax)
    ax.set_title('Pima Diabetes Dataset [Dendrogram]', **title_param)
    plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
        
def Finding_correlations_in_your_data():
    print("****************************************************")
    topic = "5. Finding correlations in your data"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------HEATMAP')
    # Plot missingness heatmap of diabetes
    fig, ax = plt.subplots(figsize=(10, 5.9))
    msno.heatmap(diabetes, fontsize=8, ax=ax)
    ax.set_title('Pima Diabetes Dataset [Heatmap]', **title_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
        
    print('---------------------------------------------DENDROGRAM')
    # Plot missingness dendrogram of diabetes
    fig, ax = plt.subplots(figsize=(10, 5.9))
    msno.dendrogram(diabetes, fontsize=8, ax=ax)
    ax.set_title('Pima Diabetes Dataset [Dendrogram]', **title_param)
    plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
     
def Visualizing_missingness_across_a_variable():
    print("****************************************************")
    topic = "7. Visualizing missingness across a variable"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    # Create dummy dataframe
    diabetes_dummy = fill_dummy_values(diabetes)
    print(diabetes_dummy.head())

    # Get missing values of both columns for coloring
    nullity = diabetes.Serum_Insulin.isnull() | diabetes.BMI.isnull()
    diabetes_dummy['category'] = nullity.replace({True: 'Null values', False: 'Not null values'})
    nullity2 = diabetes.Serum_Insulin.isnull()*1 + diabetes.BMI.isnull()*1
    diabetes_dummy['category2'] = nullity2.replace({0: 'Not null variables', 1: 'One null variable', 2: 'Both null variables'})
         
    print('---------------------------------------------MAKING THE SCATTERPLOT (SEABORN 1NULL)')
    # Generate scatter plot
    fig, ax = plt.subplots(figsize=(10, 5.9))
    sns.scatterplot(data=diabetes_dummy, hue='category', x='Serum_Insulin', y='BMI', 
                    alpha=0.5, palette='husl', ax=ax)
    ax.legend()
    ax.set_title('Relation between Missing and Non Missing Values [Seaborn 1 Null Var]', **title_param)
    #plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------MAKING THE SCATTERPLOT (SEABORN 2NULL)')
    # Generate scatter plot
    fig, ax = plt.subplots(figsize=(10, 5.9))
    sns.scatterplot(data=diabetes_dummy, hue='category2', x='Serum_Insulin', y='BMI', 
                    alpha=0.5, palette='husl', ax=ax)
    ax.legend()
    ax.set_title('Relation between Missing and Non Missing Values [Seaborn 2 Null Var]', **title_param)
    #plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------MAKING THE SCATTERPLOT (PANDAS)')
    # Generate scatter plot
    fig, ax = plt.subplots(figsize=(10, 5.9))
    diabetes_dummy.set_index('Serum_Insulin').sort_index().groupby('category')['BMI'].plot(style='.', ms=15, alpha=.5, legend=True)
    #diabetes_dummy.plot(x='Serum_Insulin', y='BMI', kind='scatter', alpha=0.5,
    #                    c=nullity, cmap='rainbow', colorbar=False, ax=ax)
    ax.legend(numpoints=1)
    ax.set_xlabel('Serum_Insulin', fontsize=8)
    ax.set_ylabel('BMI', fontsize=8)
    ax.set_title('Relation between Missing and Non Missing Values [Pandas]', **title_param)
    #plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------MAKING THE SCATTERPLOT (MATPLOTLIB)')
    # Generate scatter plot
    fig, ax = plt.subplots(figsize=(10, 5.9))
    for label, df in diabetes_dummy.groupby('category'):
        ax.scatter(x=df['Serum_Insulin'], y=df['BMI'], alpha=0.5, cmap='rainbow', label=label)
    #diabetes_dummy.plot(x='Serum_Insulin', y='BMI', kind='scatter', alpha=0.5,
    #                    c=nullity, cmap='rainbow', colorbar=False, ax=ax)
    ax.legend(numpoints=1)
    ax.set_xlabel('Serum_Insulin', fontsize=8)
    ax.set_ylabel('BMI', fontsize=8)
    ax.set_title('Relation between Missing and Non Missing Values [Matplotlib]', **title_param)
    #plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------MAKING THE SCATTERPLOT (MATPLOT 1NULL)')
    # Generate scatter plot
    fig, ax = plt.subplots(figsize=(10, 5.9))
    scat = ax.scatter(x=diabetes_dummy['Serum_Insulin'], y=diabetes_dummy['BMI'], 
                      c=nullity, alpha=0.5, cmap='rainbow')
    ax.set_xlabel('Serum_Insulin', fontsize=8)
    ax.set_ylabel('BMI', fontsize=8)
    # produce a legend with the unique colors from the scatter
    #ax.legend(*scat.legend_elements())
    ax.legend(scat.legend_elements()[0], ['Not null values', 'Null values'])
    ax.set_title('Relation between Missing and Non Missing Values [Matplotlib 1 Null Var]', **title_param)
    #plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------MAKING THE SCATTERPLOT (MATPLOT 2NULL)')
    # Generate scatter plot
    fig, ax = plt.subplots(figsize=(10, 5.9))
    labels, index = np.unique(diabetes_dummy.category2, return_inverse=True)
    scat = ax.scatter(x=diabetes_dummy['Serum_Insulin'], y=diabetes_dummy['BMI'], 
                      c=index, alpha=0.5, cmap='rainbow')
    ax.set_xlabel('Serum_Insulin', fontsize=8)
    ax.set_ylabel('BMI', fontsize=8)
    # produce a legend with the unique colors from the scatter
    #ax.legend(scat.legend_elements()[0], ['Not null values', 'One null variable', 'Both null values'])
    ax.legend(scat.legend_elements()[0], labels)
    ax.set_title('Relation between Missing and Non Missing Values [Matplotlib 2 Null Var]', **title_param)
    #plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def Generate_scatter_plot_with_missingness():
    print("****************************************************")
    topic = "9. Generate scatter plot with missingness"; print("** %s" % topic)
    print("****************************************************")
    
    # Fill dummy values in diabetes_dummy
    diabetes_dummy = fill_dummy_values(diabetes)
    # Sum the nullity of Skin_Fold and BMI
    nullity = diabetes.BMI.isnull() | diabetes.Skin_Fold.isnull()
    
    fig, ax = plt.subplots(figsize=(10, 5.9))
    # Create a scatter plot of Skin Fold and BMI 
    diabetes_dummy.plot(x='Skin_Fold', y='BMI', kind='scatter', alpha=0.5, 
                        c=nullity, cmap='rainbow', ax=ax)
    ax.set_title('Relation between Missing and Non Missing Values', **title_param)
    plt.subplots_adjust(left=None, bottom=None, right=1, top=None, wspace=None, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
        
def Delete_MCAR():
    print("****************************************************")
    topic = "11. Delete MCAR"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------VISUALIZE MISSING VALUES')
    fig, axis = plt.subplots(1,2, figsize=(12.1,4))
    ax = axis[0]
    ax = msno.matrix(diabetes, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset before Glucosa List-Wise Deletion', **title_param)
    
    print('---------------------------------------------COUNTING MISSING VALUES IN GLUCOSA COLUMN')
    # Print the number of missing values in Glucose
    print("Number of missing values: ", diabetes.Glucose.isna().sum(), '\n')
    
    print('---------------------------------------------LIST-WISE DELETIONS APPLIED TO GLUCOSA COLUMN')
    # Drop rows where 'Glucose' has a missing value
    diabetes.dropna(subset=['Glucose'], how='any', inplace=True)
    
    print('---------------------------------------------VISUALIZE UPDATED MISSING VALUES')
    ax = axis[1]
    ax = msno.matrix(diabetes, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset after Glucosa List-Wise Deletion', **title_param)
    
    
    plt.subplots_adjust(left=None, bottom=.05, right=None, top=.65, wspace=None, hspace=None); #To set the margins 
    plt.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------VISUALIZE (ALONE) UPDATED MISSING VALUES')
    #fig, ax = plt.subplots()
    ax = msno.matrix(diabetes, sparkline=True, fontsize=8, figsize=(10, 5.9)) #ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset after Glucosa List-Wise Deletion', **title_param)
    plt.subplots_adjust(left=None, bottom=.05, right=None, top=.76, wspace=None, hspace=None); #To set the margins 
    plt.suptitle(topic, **suptitle_param)
    plt.show()
    
        
def Will_you_delete():
    print("****************************************************")
    topic = "12. Will you delete?"; print("** %s" % topic)
    print("****************************************************")
    
    suptitle_param   = dict(color='darkblue', fontsize=9)
    title_param      = {'color': 'darkred', 'fontsize': 8}
    
    fig, axis = plt.subplots(2, 3, figsize=(12.1, 5.9))
    print('---------------------------------------------MATRIX ')
    ax = axis[0,0]
    msno.matrix(diabetes, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset - Matrix [BEFORE DELETION]]', **title_param)
    
    print('---------------------------------------------HEATMAP')
    ax = axis[0,1]
    msno.heatmap(diabetes, fontsize=8, ax=ax)
    ax.set_title('Pima Diabetes Dataset - Heatmap [BEFORE DELETION]', **title_param)
    
    print('---------------------------------------------DENDROGRAM')
    ax = axis[0,2]
    msno.dendrogram(diabetes, fontsize=8, ax=ax)
    ax.set_title('Pima Diabetes Dataset - Dendrogram [BEFORE DELETION]', **title_param)
    
    print('---------------------------------------------LIST-WISE DELETIONS APPLIED TO BMI COLUMN')
    # Drop rows where 'BMI' has a missing value
    diabetes.dropna(subset=['BMI'], how='all', inplace=True)

    print('---------------------------------------------MATRIX ')
    ax = axis[1,0]
    msno.matrix(diabetes, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset - Matrix [AFTER DELETION]', **title_param)
    
    print('---------------------------------------------HEATMAP')
    ax = axis[1,1]
    msno.heatmap(diabetes, fontsize=8, ax=ax)
    ax.set_title('Pima Diabetes Dataset - Heatmap [AFTER DELETION]', **title_param)
    
    print('---------------------------------------------DENDROGRAM')
    ax = axis[1,2]
    msno.dendrogram(diabetes, fontsize=8, ax=ax)
    ax.set_title('Pima Diabetes Dataset - Dendrogram [AFTER DELETION]', **title_param)
    
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.75, wspace=.5, hspace=1.5); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Is_the_data_missing_at_random()
    Guess_the_missingness_type()
    Deduce_MNAR()
    Finding_patterns_in_missing_data()
    Finding_correlations_in_your_data()
    Visualizing_missingness_across_a_variable()
    Generate_scatter_plot_with_missingness()
    Delete_MCAR()
    Will_you_delete()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()
    pd.options.display.float_format = None
    plt.style.use('default')