# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: Imputation Techniques
    Embark on the world of data imputation! In this chapter, you will apply basic 
    imputation techniques to fill in missing data and visualize your imputations to 
    be able to evaluate your imputations' performance.
Source: https://learn.datacamp.com/courses/data-manipulation-with-pandas
Help:
    https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
"""
###############################################################################
## Importing libraries
###############################################################################
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import StandardScaler


###############################################################################
## Preparing the environment
###############################################################################
# Global variables
SEED = 42 
np.random.seed(SEED)

# Global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 6, 'font.size': 8})
suptitle_param   = dict(color='darkblue', fontsize=9)
title_param      = {'color': 'darkred', 'fontsize': 10}

# Reading the data
diabetes = pd.read_csv('pima-indians-diabetes data.csv')
airquality = pd.read_csv('air-quality.csv', parse_dates=['Date'], index_col='Date')

    

###############################################################################
## Main part of the code
###############################################################################
def Mean_median_mode_imputations():
    print("****************************************************")
    topic = "1. Mean, median & mode imputations"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    print(f"diabetes Dataset's Head:\n{diabetes.head()}")
    
    print('---------------------------------------------COUNTING MISSING VALUES')
    counting_missing = diabetes.isna().sum().sort_values(ascending=False)
    columns_with_missing = counting_missing[counting_missing>0].index.tolist()
    print(counting_missing)
    print(f"\nJust columns with missing values:\n{diabetes[columns_with_missing].tail()}")
    
    print('---------------------------------------------VISUALIZE THE MISSINGNESS')
    fig, axis = plt.subplots(1, 4, figsize=(12.1, 5.9))
    ax = axis[0]
    msno.bar(diabetes, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Missing values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset [Nullity Bar]', **title_param)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax = axis[1]
    msno.matrix(diabetes, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Pima Diabetes Dataset [Matrix]', **title_param)
    ax = axis[2]
    msno.heatmap(diabetes, fontsize=8, ax=ax)
    ax.set_title('Pima Diabetes Dataset [Heatmap]', **title_param)
    ax = axis[3]
    msno.dendrogram(diabetes, fontsize=8, ax=ax)
    ax.set_title('Pima Diabetes Dataset [Dendrogram]', **title_param)
    
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.75, wspace=.5, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------MEAN IMPUTATION')
    diabetes_mean = diabetes.copy(deep=True)
    mean_imputer = SimpleImputer(strategy='mean')
    diabetes_mean.iloc[:, :] = mean_imputer.fit_transform(diabetes_mean)
    print(f"diabetes_mean Dataset's Head:\n{diabetes_mean[columns_with_missing].tail()}")
    
    print('---------------------------------------------MEDIAN IMPUTATION')
    diabetes_median = diabetes.copy(deep=True)
    median_imputer = SimpleImputer(strategy='median')
    diabetes_median.iloc[:, :] = median_imputer.fit_transform(diabetes_median)
    print(f"diabetes_median Dataset's Head:\n{diabetes_median[columns_with_missing].tail()}")
    
    print('---------------------------------------------MODE IMPUTATION')
    diabetes_mode = diabetes.copy(deep=True)
    mode_imputer = SimpleImputer(strategy='most_frequent')
    diabetes_mode.iloc[:, :] = mode_imputer.fit_transform(diabetes_mode)
    print(f"diabetes_mode Dataset's Head:\n{diabetes_mode[columns_with_missing].tail()}")
    
    print('---------------------------------------------CONSTANT IMPUTATION')
    diabetes_constant = diabetes.copy(deep=True)
    constant_imputer = SimpleImputer(strategy='constant', fill_value=0)
    diabetes_constant.iloc[:, :] = constant_imputer.fit_transform(diabetes_constant)
    print(f"diabetes_constant Dataset's Head:\n{diabetes_constant[columns_with_missing].tail()}")
    
    print('---------------------------------------------PREPARING THE SCATTER PLOT')
    nullity = diabetes['Serum_Insulin'].isnull() | diabetes['Glucose'].isnull()
    imputations = {'Mean Imputation'         : diabetes_mean,
                   'Median Imputation'       : diabetes_median,
                   'Most Frequent Imputation': diabetes_mode,
                   'Constant Imputation'     : diabetes_constant}
    
    print('---------------------------------------------SCATTER PLOT OF IMPUTATION')
    fig, axis = plt.subplots(2, 2, figsize=(12.1, 5.9))
    for ax, df_key in zip(axis.flatten(), imputations):
        imputations[df_key].plot(x='Serum_Insulin', y='Glucose', kind='scatter',
                                 alpha=0.5, c=nullity, cmap='rainbow', ax=ax,
                                 colorbar=False)
        ax.set_title(df_key, **title_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5); #To set the margins 
    fig.suptitle(f"{topic}\n(Red points represent imputation)", **suptitle_param)
    plt.show()
    
    
        
def Mean_median_imputations():
    print("****************************************************")
    topic = "2. Mean & median imputation"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------MEAN IMPUTATION')
    # Make a copy of diabetes
    diabetes_mean = diabetes.copy(deep=True)
    # Create mean imputer object
    mean_imputer = SimpleImputer(strategy='mean')
    # Impute mean values in the DataFrame diabetes_mean
    diabetes_mean.iloc[:, :] = mean_imputer.fit_transform(diabetes_mean)
    
    print('---------------------------------------------MEDIAN IMPUTATION')
    # Make a copy of diabetes
    diabetes_median = diabetes.copy(deep=True)
    # Create median imputer object
    median_imputer = SimpleImputer(strategy='median')
    # Impute median values in the DataFrame diabetes_median
    diabetes_median.iloc[:, :] = median_imputer.fit_transform(diabetes_median)
    
    return diabetes_mean, diabetes_median
    
            
def Mode_and_constant_imputation():
    print("****************************************************")
    topic = "3. Mode and constant imputation"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------MODE IMPUTATION')
    # Make a copy of diabetes
    diabetes_mode = diabetes.copy(deep=True)
    # Create mode imputer object
    mode_imputer = SimpleImputer(strategy='most_frequent')
    # Impute using most frequent value in the DataFrame mode_imputer
    diabetes_mode.iloc[:, :] = mode_imputer.fit_transform(diabetes_mode)
    
    print('---------------------------------------------CONSTANT IMPUTATION')
    # Make a copy of diabetes
    diabetes_constant = diabetes.copy(deep=True)
    # Create median imputer object
    constant_imputer = SimpleImputer(strategy='constant', fill_value=0)
    # Impute missing values to 0 in diabetes_constant
    diabetes_constant.iloc[:, :] = constant_imputer.fit_transform(diabetes_constant)
    
    return diabetes_mode, diabetes_constant
    
    
        
def Visualize_imputations(diabetes_mean, diabetes_median, diabetes_mode, diabetes_constant):
    print("****************************************************")
    topic = "4. Visualize imputations"; print("** %s" % topic)
    print("****************************************************")
    
    # Set nrows and ncols to 2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.1, 5.9))
    nullity = diabetes.Serum_Insulin.isnull() | diabetes.Glucose.isnull()
    
    # Create a dictionary of imputations
    imputations = {'Mean Imputation': diabetes_mean, 'Median Imputation': diabetes_median, 
                   'Most Frequent Imputation': diabetes_mode, 'Constant Imputation': diabetes_constant}
    
    # Loop over flattened axes and imputations
    for ax, df_key in zip(axes.flatten(), imputations):
        # Select and also set the title for a DataFrame
        imputations[df_key].plot(x='Serum_Insulin', y='Glucose', kind='scatter', 
                                 alpha=0.5, c=nullity, cmap='rainbow', ax=ax, 
                                 colorbar=False)
        ax.set_title(df_key, **title_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5); #To set the margins 
    fig.suptitle(f"{topic}\n(Red points represent imputation)", **suptitle_param)
    plt.show()
    
    
        
def Imputing_time_series_data():
    print("****************************************************")
    topic = "5. Imputing time-series data"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    print(f"Part of airquality Dataset's:\n{airquality[30:40]}")
    
    print('---------------------------------------------VISUALIZE THE MISSINGNESS')
    fig, axis = plt.subplots(1, 4, figsize=(12.1, 5.9))
    ax = axis[0]
    msno.bar(airquality, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Missing values', fontsize=8)
    ax.set_title('Airquality Dataset [Nullity Bar]', **title_param)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax = axis[1]
    msno.matrix(airquality, sparkline=False, fontsize=8, ax=ax)
    ax.set_xlabel('Variables', fontsize=8)
    ax.set_ylabel('Number of values', fontsize=8)
    ax.set_title('Airquality Dataset [Matrix]', **title_param)
    ax = axis[2]
    msno.heatmap(airquality, fontsize=8, ax=ax)
    ax.set_title('Airquality Dataset [Heatmap]', **title_param)
    ax = axis[3]
    msno.dendrogram(airquality, fontsize=8, ax=ax)
    ax.set_title('Airquality Dataset [Dendrogram]', **title_param)
    
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.75, wspace=.5, hspace=None); #To set the margins 
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    print('---------------------------------------------FINDING MISSING VALUES PER COLUMN')
    counting_missing = airquality.isna().sum().sort_values(ascending=False)
    columns_with_missing = counting_missing[counting_missing>0].index.tolist()
    print(counting_missing, '\n')
    
    print('---------------------------------------------PERCENTAGE OF MISSINGNESS')
    print(airquality.isna().mean().sort_values(ascending=False) * 100, '\n')
    
    print('---------------------------------------------APPLYING .fillna METHOD')
    fillna_method = ['ffill', 'bfill']
    for column_name in columns_with_missing:
        fig, axes = plt.subplots(len(fillna_method), 1, figsize=(12.1, 5.9))
        
        for ax, method_to_apply in zip(axes.flatten(), fillna_method):
            airquality_fillna = airquality.fillna(method=method_to_apply)
            subtopic = f".fillna METHOD WITH ***{method_to_apply}*** METHOD"
            print(f"{subtopic}\nPart of airquality_fillna Dataset's:\n{airquality_fillna[30:40]}\n")
            
            airquality_fillna['category'] = np.where(airquality[column_name].isnull(), 'Null values', 'Not null values')
            airquality_fillna.plot(use_index=True, y=column_name, lw=1, ax=ax)
            airquality_fillna.sort_index().groupby('category')[column_name].plot(style='.', legend=True, ax=ax)
            ax.set_xlabel(' ')
            ax.legend(loc='upper right')
            ax.axhline(airquality[column_name].max(), color='darkblue', ls='--', lw=.5)
            ax.axhline(airquality[column_name].min(), color='darkblue', ls='--', lw=.5)
            ax.set_title(f"{column_name}\n{subtopic}", **title_param)
        fig.suptitle(topic, **suptitle_param)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5); #To set the margins 
        plt.show()
    
    print('---------------------------------------------APPLYING .interpolate METHOD')
    interpolate_method = ['linear', 'quadratic', 'nearest']
    for column_name in columns_with_missing:
        fig, axes = plt.subplots(len(interpolate_method), 1, figsize=(12.1, 5.9))
        
        for ax, method_to_apply in zip(axes.flatten(), interpolate_method):
            airquality_interpolate = airquality.interpolate(method=method_to_apply)
            subtopic = f".interpolate METHOD WITH ***{method_to_apply}*** METHOD"
            print(f"{subtopic}\nPart of airquality_interpolate Dataset's:\n{airquality_interpolate[30:40]}\n")
            
            airquality_interpolate['category'] = np.where(airquality[column_name].isnull(), 'Null values', 'Not null values')
            airquality_interpolate.plot(use_index=True, y=column_name, lw=1, ax=ax)
            airquality_interpolate.sort_index().groupby('category')[column_name].plot(style='.', legend=True, ax=ax)
            ax.axhline(airquality[column_name].max(), color='darkblue', ls='--', lw=.5)
            ax.axhline(airquality[column_name].min(), color='darkblue', ls='--', lw=.5)
            ax.set_xlabel(' ')
            ax.legend(loc='upper right')
            ax.set_title(f"{column_name}\n{subtopic}", **title_param)
        fig.suptitle(topic, **suptitle_param)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.9); #To set the margins 
        plt.show()
    
    
        
def Filling_missing_time_series_data():
    print("****************************************************")
    topic = "6. Filling missing time-series data"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    # Print prior to imputing missing values
    print(airquality[30:40])
    
    print('---------------------------------------------FILLNA WITH FFILL METHOD')
    # Fill NaNs using forward fill
    airquality_fillna = airquality.fillna(method='ffill')
    # Print after imputing missing values
    print(airquality_fillna [30:40])
    
    print('---------------------------------------------FILLNA WITH BFILL METHOD')
    # Fill NaNs using backward fill
    airquality_fillna = airquality.fillna(method='bfill')
    # Print after imputing missing values
    print(airquality_fillna[30:40])
    
    
    
def Impute_with_interpolate_method():
    print("****************************************************")
    topic = "7. Impute with interpolate method"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------EXPLORE')
    # Print prior to imputing missing values
    print(airquality[30:40])
    
    print('---------------------------------------------INTERPOLATION WITH LINEAR METHOD')
    # Interpolate the NaNs linearly
    airquality_interpolate = airquality.interpolate(method='linear')
    # Print after interpolation
    print(airquality_interpolate[30:40])
    
    print('---------------------------------------------INTERPOLATION WITH QUADRATIC METHOD')
    # Interpolate the NaNs quadratically
    airquality_interpolate = airquality.interpolate(method='quadratic')
    # Print after interpolation
    print(airquality_interpolate[30:40])
    
    print('---------------------------------------------INTERPOLATION WITH NEAREST METHOD')
    # Interpolate the NaNs with nearest value
    airquality_interpolate = airquality.interpolate(method='nearest')
    # Print after interpolation
    print(airquality_interpolate[30:40])
    
    
    
def Visualizing_time_series_imputations():
    print("****************************************************")
    topic = "8. Visualizing time-series imputations"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------FINDING MISSING VALUES PER COLUMN')
    counting_missing = airquality.isna().sum().sort_values(ascending=False)
    columns_with_missing = counting_missing[counting_missing>0].index.tolist()
    print(counting_missing, '\n')
    
    print('---------------------------------------------APPLYING .fillna METHOD')
    airquality_fillna_ffill = airquality.fillna(method='ffill')
    airquality_fillna_bfill = airquality.fillna(method='bfill')
    
    print('---------------------------------------------APPLYING .interpolate METHOD')
    airquality_interpolate_linear = airquality.interpolate(method='linear')
    airquality_interpolate_quadratic = airquality.interpolate(method='quadratic')
    airquality_interpolate_nearest = airquality.interpolate(method='nearest')
            
    print('---------------------------------------------COMPARE THE METHODS')
    # Create interpolations dictionary
    interpolations = {'Original Data'          : airquality,
                      'Quadratic Interpolation': airquality_interpolate_quadratic,
                      'Forward fill'           : airquality_fillna_ffill,
                      'Linear Interpolation'   : airquality_interpolate_linear,
                      'Backward fill'          : airquality_fillna_bfill,
                      'Nearest Interpolation'  : airquality_interpolate_nearest}
    
    # Visualize each interpolation
    for column_name in columns_with_missing:
        # Create subplots    
        fig, axes = plt.subplots(3, 2, figsize=(12.1, 5.9))
        
        for ax, df_key in zip(axes.flatten(), interpolations):
            interpolations[df_key][column_name].plot(color='red', marker='o', linestyle='dotted', lw=.75, ms=1.5, ax=ax)
            airquality[column_name].plot(color='blue', marker='o', lw=1, ms=3, ax=ax)
            #ax.set_ylabel(column_name)
            ax.set_xlabel(' ')
            ax.axhline(airquality[column_name].max(), color='gray', ls='--', lw=.5)
            ax.axhline(airquality[column_name].min(), color='gray', ls='--', lw=.5)
            ax.set_title(f"Column: '{column_name.upper()}' - {df_key}", **title_param)
        fig.suptitle(topic, **suptitle_param)
        plt.subplots_adjust(left=.05, bottom=None, right=.95, top=None, wspace=.2, hspace=.5); #To set the margins 
        plt.show()
        
    return airquality_interpolate_linear, airquality_interpolate_quadratic, airquality_interpolate_nearest
    
    
        
def Visualize_forward_fill_imputation():
    print("****************************************************")
    topic = "9. Visualize forward fill imputation"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------PLOTTING FILLNA WITH FFILL METHOD')
    column_name = 'Ozone'
    df_key = 'Forward fill imputation'
    
    # Impute airquality DataFrame with ffill method
    ffill_imputed = airquality.fillna(method='ffill')
    
    fig, ax = plt.subplots(figsize=(12.1, 5.9))
    # Plot the imputed DataFrame ffill_imp in red dotted style 
    ffill_imputed[column_name].plot(color='red', marker='o', linestyle=':', lw=1, ms=2, label='Imputation data', ax=ax)
    # Plot the airquality DataFrame with title
    airquality[column_name].plot(marker='o', lw=1, ms=2, label='Original data', ax=ax)
    ax.axhline(airquality[column_name].max(), color='gray', ls='--', lw=.5)
    ax.axhline(airquality[column_name].min(), color='gray', ls='--', lw=.5)
    ax.set_xlabel(' ')
    ax.legend()
    ax.set_title(f"Column: '{column_name.upper()}' - {df_key}", **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.05, bottom=None, right=.95, top=None, wspace=.2, hspace=.5); #To set the margins 
    plt.show()
    
    
        
def Visualize_backward_fill_imputation():
    print("****************************************************")
    topic = "10. Visualize backward fill imputation"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------PLOTTING FILLNA WITH BFILL METHOD')
    column_name = 'Ozone'
    df_key = 'Backward fill imputation'
    
    fig, ax = plt.subplots(figsize=(12.1, 5.9))
    # Impute airquality DataFrame with bfill method
    bfill_imputed = airquality.fillna(method='bfill')
    # Plot the imputed DataFrame bfill_imp in red dotted style 
    bfill_imputed[column_name].plot(color='red', marker='o', linestyle='dotted', lw=1, ms=2, label='Imputation data', ax=ax)
    # Plot the airquality DataFrame with title
    airquality[column_name].plot(marker='o', label='Original data', lw=1, ms=2, ax=ax)
    ax.axhline(airquality[column_name].max(), color='gray', ls='--', lw=.5)
    ax.axhline(airquality[column_name].min(), color='gray', ls='--', lw=.5)
    ax.set_xlabel(' ')
    ax.legend()
    ax.set_title(f"Column: '{column_name.upper()}' - {df_key}", **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.05, bottom=None, right=.95, top=None, wspace=.2, hspace=.5); #To set the margins 
    plt.show()
    
    
        
def Plot_interpolations(linear, quadratic, nearest):
    print("****************************************************")
    topic = "11. Plot interpolations"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------PLOTTING INTERPOLATE METHOD')
    # Create a dictionary of interpolations
    column_name = 'Ozone'
    interpolations = {'Linear Interpolation': linear, 
                      'Quadratic Interpolation': quadratic, 
                      'Nearest Interpolation': nearest}

    # Set nrows to 3 and ncols to 1
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12.1, 5.9))
    # Loop over axes and interpolations
    for ax, df_key in zip(axes, interpolations):
            # Select and also set the title for a DataFrame
        interpolations[df_key][column_name].plot(color='red', marker='o', linestyle=':', lw=1, ms=2, label='Imputation data', ax=ax)
        airquality[column_name].plot(marker='o', label='Original data', lw=1, ms=2, ax=ax)
        ax.axhline(airquality[column_name].max(), color='gray', ls='--', lw=.5)
        ax.axhline(airquality[column_name].min(), color='gray', ls='--', lw=.5)
        ax.set_xlabel(' ')
        ax.legend()
        ax.set_title(df_key + ' - ' + column_name, **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.9); #To set the margins 
    plt.show()
    
    
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Mean_median_mode_imputations()
    diabetes_mean, diabetes_median = Mean_median_imputations()
    diabetes_mode, diabetes_constant = Mode_and_constant_imputation()
    Visualize_imputations(diabetes_mean, diabetes_median, diabetes_mode, diabetes_constant)
    Imputing_time_series_data()
    Filling_missing_time_series_data()
    Impute_with_interpolate_method()
    linear, quadratic, nearest = Visualizing_time_series_imputations()
    Visualize_forward_fill_imputation()
    Visualize_backward_fill_imputation()
    Plot_interpolations(linear, quadratic, nearest)
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')