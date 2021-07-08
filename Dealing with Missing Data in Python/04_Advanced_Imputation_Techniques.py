# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Advanced Imputation Techniques
    Finally, go beyond simple imputation techniques and make the most of your 
    dataset by using advanced imputation techniques that rely on machine learning 
    models, to be able to accurately impute and evaluate your missing data. You 
    will be using methods such as KNN and MICE in order to get the most out of 
    your missing data!
Source: https://learn.datacamp.com/courses/data-manipulation-with-pandas
Help:
    https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
"""
###############################################################################
## Importing libraries
###############################################################################
import matplotlib.pyplot as plt
#import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from fancyimpute import KNN
from fancyimpute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer



###############################################################################
## Preparing the environment
###############################################################################
# Global variables
SEED = 42 
np.random.seed(SEED)

# Global configuration
pd.set_option("display.max_columns", 20)
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 6, 'font.size': 8})
suptitle_param   = dict(color='darkblue', fontsize=9)
title_param      = {'color': 'darkred', 'fontsize': 10}


# Global functions
def compare_imputations(columns_with_missing, imputations, topic, ncols=1, figsize=(12.1, 5.9), title_param=title_param, suptitle_param=suptitle_param):
    """
    Shows a comparative graph of the differents imputations prepared. The base of the 
    function is imputations param, that has the following stryctyre:
        imputations = {'Original Data': original_df,
                       'Method 1'     : imputation_df_1,
                       ...
                       'Method n'     : imputation_df_n}
    Parameters
    ----------
    columns_with_missing : List of string, containing the names of the columns with missing values.
    imputations          : Dictionary of the original data + imputations.
    ncols                : Number of columns to subplots
    figsize              : Size of the graph. The default is (12.1, 5.9).
    title_param          : Dict of the specific params to format the title of the plot.
    suptitle_param       : Dict of the specific params to format the suptitle of the plot.
    """
    # Defining the number of rows to subplot
    nrows = round(len(imputations)/ncols)
    original_title = list(imputations.items())[0][0]
    original_data = list(imputations.items())[0][1]
    
    # Visualize each imputations
    for column_name in columns_with_missing:
        # Create subplots    
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        for i, (ax, df_key) in enumerate(zip(axes.flatten(), imputations)):
            if i>0:
                imputations[df_key][column_name].plot(color='red', marker='o', linestyle='dotted', lw=.5, ms=1, label='Imputed data', ax=ax)
                ax.axhspan(original_data[column_name].min(), original_data[column_name].max(), color='antiquewhite', label='range', lw=.5)
            original_data[column_name].plot(color='blue', marker='o', lw=.5, ms=1, label=original_title, ax=ax)
            ax.set_xlabel(' ')
            ax.legend(loc='upper right')
            ax.set_title(f"Column: '{column_name.upper()}' - {df_key}", **title_param)
        fig.suptitle(topic, **suptitle_param)
        plt.subplots_adjust(left=.05, bottom=None, right=.95, top=None, wspace=.2, hspace=.5); #To set the margins 
        if i < (nrows*ncols)-1: axes.flatten()[i+1].axis('off')
        plt.show()
        
def Apply_different_imputations_models(df):
    """
    Make different kind of imputations (Mean, Median, Mode, 0 Value, KNN and MICE).
    This method is for nod time serias dataframes.
    Parameters
    ----------
    df: Dataframe object with null values.
    Returns
    -------
    columns_with_missing : List of column name that had null values.
    imputations_dict     : Dictionary of dataframes created with the different imputations methods.
                           The dictionary has the following structure: 
                               imputations = {'Original Data': original_df,
                                              'Method 1'     : imputation_df_1,
                                              ...
                                              'Method n'     : imputation_df_n}
    """
    #---------------------------------------------Finding columns with missing values
    counting_missing = diabetes.isna().sum().sort_values(ascending=False)
    columns_with_missing = counting_missing[counting_missing>0].index.tolist()
    
    #---------------------------------------------SIMPLE IMPUTATION
    method_imputers = {'mean imputer'    : SimpleImputer(strategy='mean'), 
                       'median imputer'  : SimpleImputer(strategy='median'),
                       'mode imputer'    : SimpleImputer(strategy='most_frequent'), 
                       'constant imputer': SimpleImputer(strategy='constant', fill_value=0),
                       'KNN imputer'     : KNN(verbose=False),
                       'MICE imputer'    : IterativeImputer()}
    imputations_dict = {'original data': df}
    for method in method_imputers:
        df_simple_imputer = df.copy(deep=True)
        simple_imputer = method_imputers[method]
        df_simple_imputer.iloc[:, :] = simple_imputer.fit_transform(df_simple_imputer)
        imputations_dict[method] = df_simple_imputer
    
    return columns_with_missing, imputations_dict

def Evaluate_imputation_model(imputations):
    """
    Evaluate different given imputations.
    Parameters
    ----------
    imputations : Dictionary of dataframes created with the different imputations methods.
                  The dictionary has the following structure: 
                      imputations = {'Original Data': original_df,
                                     'Method 1'     : imputation_df_1,
                                     ...
                                     'Method n'     : imputation_df_n}
    Returns
    -------
    df_rsquared     : Dataframe with the calculated R_squared for each given imputations. 
    df_rsquared_adj : Dataframe with the calculated R_squared_adj for each given imputations.
    df_coef         : Dataframe with the calculated coefficient for each given imputations.
    lm_dict         : Dictionary with the prepared linar regression model for each given imputations.
    """
    # Prepare the dataframes
    df_rsquared     = pd.DataFrame({})
    df_rsquared_adj = pd.DataFrame({})
    df_coef         = pd.DataFrame({})
    lm_dict         = {}
    
    for df_key in imputations:
        # Fit a linear model for statistical summary
        df_cc = imputations[df_key].dropna(how='any')
        imputations[df_key] = df_cc # Update without nulls
        X = sm.add_constant(df_cc.iloc[:, :-1])
        y = df_cc['Class']
        lm = sm.OLS(y, X).fit()
    
        # R-squared and Coef
        df_rsquared[df_key] = [lm.rsquared]
        df_rsquared_adj[df_key] = [lm.rsquared_adj]
        df_coef[df_key] = lm.params
        lm_dict[df_key] = lm
    
    df_rsquared.index     = ['R_squared']
    df_rsquared_adj.index = ['R_squared_adj']
    return df_rsquared, df_rsquared_adj, df_coef, lm_dict, imputations

def Visualize_KDE_from_imputations_model(columns_with_missing, imputations_notnulls, topic, figsize=(12.1, 5.9), title_param=title_param, suptitle_param=suptitle_param):
    """
    Visualize the KDE of the columns int the different imputations given models.
    Parameters
    ----------
    columns_with_missing : List of string, containing the names of the columns with missing values.
    imputations_notnulls : Dictionary of the original data + imputations.
    figsize              : Size of the graph. The default is (12.1, 5.9).
    title_param          : Dict of the specific params to format the title of the plot.
    suptitle_param       : Dict of the specific params to format the suptitle of the plot.
    Returns
    -------
    None.

    """
    # Visualize each imputations
    for column_name in columns_with_missing:
        # Create subplots    
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, df_key in enumerate(imputations_notnulls):
            extra_params = dict(linewidth=3) if i==0 else {}
            imputations_notnulls[df_key][column_name].plot(kind='kde', ax=ax, **extra_params)
        ax.set_xlabel(' ')
        ax.legend(imputations_notnulls.keys(), loc='upper right')
        ax.set_title(f"Column: '{column_name.upper()}'", **title_param)
        fig.suptitle(topic, **suptitle_param)
        plt.subplots_adjust(left=.05, bottom=None, right=.95, top=None, wspace=.2, hspace=.5); #To set the margins 
        plt.show()
        
        
        
###############################################################################
## Reading the data
###############################################################################
diabetes = pd.read_csv('pima-indians-diabetes data.csv')
users = pd.read_csv('userprofile_nullvalues.csv', sep=';')
#for colname in users:
#    users[colname] = users[colname].astype('category')

###############################################################################
## Main part of the code
###############################################################################
def Imputing_using_fancyimpute():
    print("****************************************************")
    topic = "1. Imputing using fancyimpute"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Explore')
    print(f"diabetes Dataset's Head:\n{diabetes.head()}")
    
    print('---------------------------------------------Counting missing values')
    counting_missing = diabetes.isna().sum().sort_values(ascending=False)
    columns_with_missing = counting_missing[counting_missing>0].index.tolist()
    print(counting_missing)
    print(f"\nJust columns with missing values:\n{diabetes[columns_with_missing].tail()}")
    
    print('---------------------------------------------K-Nearest Neighbor Imputation (KNN)')
    knn_imputer = KNN()
    diabetes_knn = diabetes.copy(deep=True)
    diabetes_knn.iloc[:, :] = knn_imputer.fit_transform(diabetes_knn)
    print(diabetes_knn[columns_with_missing].tail())
    
    print('---------------------------------------------Multiple Imputations by Chained Equations(MICE)')
    MICE_imputer = IterativeImputer()
    diabetes_mice = diabetes.copy(deep=True)
    diabetes_mice.iloc[:, :] = MICE_imputer.fit_transform(diabetes_mice)
    print(diabetes_knn[columns_with_missing].tail())
    
    print('---------------------------------------------Visualizing the imputations')
    # Create imputation dictionary
    imputations = {'Original Data'  : diabetes,
                   'KNN Imputation' : diabetes_knn,
                   'MICE Imputation': diabetes_mice}
    
    # Visualize imputations
    compare_imputations(columns_with_missing, imputations, topic)
    
    
    
def KNN_imputation():
    print("****************************************************")
    topic = "2. KNN imputation"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Counting missing values')
    counting_missing = diabetes.isna().sum().sort_values(ascending=False)
    columns_with_missing = counting_missing[counting_missing>0].index.tolist()
    
    print('---------------------------------------------K-Nearest Neighbor Imputation (KNN)')
    # Copy diabetes to diabetes_knn_imputed
    diabetes_knn_imputed = diabetes.copy(deep=True)
    
    # Initialize KNN
    knn_imputer = KNN()
    
    # Impute using fit_tranform on diabetes_knn_imputed
    diabetes_knn_imputed.iloc[:, :] = knn_imputer.fit_transform(diabetes_knn_imputed)
    
    print('---------------------------------------------Visualizing the imputations')
    imputations = {'Original Data'  : diabetes,
                   'KNN Imputation' : diabetes_knn_imputed}
    # Visualize imputations
    compare_imputations(columns_with_missing, imputations, topic)
    
    
        
def MICE_imputation():
    print("****************************************************")
    topic = "3. MICE imputation"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Counting missing values')
    counting_missing = diabetes.isna().sum().sort_values(ascending=False)
    columns_with_missing = counting_missing[counting_missing>0].index.tolist()
    
    print('---------------------------------------------Multiple Imputations by Chained Equations(MICE)')
    # Copy diabetes to diabetes_mice_imputed
    diabetes_mice_imputed = diabetes.copy(deep=True)
    
    # Initialize IterativeImputer
    mice_imputer = IterativeImputer()
    
    # Impute using fit_tranform on diabetes
    diabetes_mice_imputed.iloc[:, :] = mice_imputer.fit_transform(diabetes_mice_imputed)
    
    print('---------------------------------------------Visualizing the imputations')
    imputations = {'Original Data'  : diabetes,
                   'MICE Imputation' : diabetes_mice_imputed}
    # Visualize imputations
    compare_imputations(columns_with_missing, imputations, topic)
    
    
        
def Imputing_categorical_values():
    print("****************************************************")
    topic = "4. Imputing categorical values"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Explore')
    print(f"Users Dataset's Head:\n{users.head()}")
    
    print('---------------------------------------------Counting missing values')
    counting_missing = users.isna().sum().sort_values(ascending=False)
    columns_with_missing = counting_missing[counting_missing>0].index.tolist()
    print(counting_missing)
    print(f"\nJust columns with missing values:\n{users.loc[22:27, columns_with_missing]}")
    
    print('---------------------------------------------Visualize categorical plots')
    users_dummy = users.fillna('null values')
    fig, axes = plt.subplots(3, 2, figsize=(12.1, 5.9))
    for ax, column_name in zip(axes.flatten(), columns_with_missing):
        order = np.append(users[column_name].value_counts().index, 'null values')
        sns.countplot(x=column_name, data=users_dummy, 
                      palette = np.append(np.repeat('blue', users[column_name].nunique()), 'red'),
                      order=order, ax=ax)
        ax.set_xlabel('Unique values')
        ax.set_ylabel('Frequency')
        ax.set_title(f"Column: '{column_name.upper()}'", **title_param)
    fig.suptitle(f"{topic}\nBEFORE IMPUTATIONS", **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.7); #To set the margins 
    plt.show()
    
    print('---------------------------------------------Ordinal Encoding')
    users_encoder = users.copy(deep=True)
    # Create dictionary for Ordinal encoders
    ordinal_enc_dict = {}
    # Loop over columns to encode
    for col_name in users_encoder:
        # Create ordinal encoder for the column
        ordinal_enc_dict[col_name] = OrdinalEncoder()
        # Select the nin-null values in the column
        col = users_encoder[col_name]
        col_not_null = col[col.notnull()]
        reshaped_vals = col_not_null.values.reshape(-1, 1)
        # Encode the non-null values of the column
        encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)
        # Replace the column with ordinal values
        users_encoder.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
    print(users_encoder.loc[22:27])
    
    print('---------------------------------------------Imputing with KNN')
    users_KNN_imputed = users_encoder.copy(deep=True)
    # Create MICE imputer
    KNN_imputer = KNN()
    # Impute the missing data
    users_KNN_imputed.iloc[:, :] = np.round(KNN_imputer.fit_transform(users_KNN_imputed))
    print(users_KNN_imputed.loc[22:27])
    
    
    print('---------------------------------------------Recode the data')
    for col_name in users_KNN_imputed:
        reshaped_col = users_KNN_imputed[col_name].values.reshape(-1, 1)
        users_KNN_imputed[col_name] = ordinal_enc_dict[col_name].inverse_transform(reshaped_col)
    print(users_KNN_imputed.loc[22:27])
    
    print('---------------------------------------------Counting missing values after imputations')
    print(users_KNN_imputed.isna().sum().sort_values(ascending=False))
    
    print('---------------------------------------------Visualize categorical plots after imputations')
    #users_dummy = users.fillna('null values')
    fig, axes = plt.subplots(3, 2, figsize=(12.1, 5.9))
    for ax, column_name in zip(axes.flatten(), columns_with_missing):
        order = np.append(users[column_name].value_counts().index, 'null values')
        sns.countplot(x=column_name, data=users_KNN_imputed, 
                      palette = np.append(np.repeat('blue', users[column_name].nunique()), 'red'),
                      order=order, ax=ax)
        ax.set_xlabel('Unique values')
        ax.set_ylabel('Frequency')
        ax.set_title(f"Column: '{column_name.upper()}'", **title_param)
    fig.suptitle(f"{topic}\nAFTER IMPUTATIONS", **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.7); #To set the margins 
    plt.show()
    
    
        
def Ordinal_encoding_of_a_categorical_column(users):
    print("****************************************************")
    topic = "5. Ordinal encoding of a categorical column"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Explore')
    print(users.loc[22:27, 'ambience'])
    
    print('---------------------------------------------Encode the column')
    # Create Ordinal encoder
    ambience_ord_enc = OrdinalEncoder()    
    # Select non-null values of ambience column in users
    ambience = users['ambience']
    ambience_not_null = ambience[ambience.notnull()]    
    # Reshape ambience_not_null to shape (-1, 1)
    reshaped_vals = ambience_not_null.values.reshape(-1,1)    
    # Ordinally encode reshaped_vals
    encoded_vals = ambience_ord_enc.fit_transform(reshaped_vals)    
    # Assign back encoded values to non-null values of ambience in users
    users.loc[ambience.notnull(), 'ambience'] = np.squeeze(encoded_vals)
    print(users.loc[22:27, 'ambience'])
    
        
def Ordinal_encoding_of_a_DataFrame():
    print("****************************************************")
    topic = "6. Ordinal encoding of a DataFrame"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Counting missing values')
    counting_missing = users.isna().sum().sort_values(ascending=False)
    columns_with_missing = counting_missing[counting_missing>0].index.tolist()
    print(counting_missing)
    print(f"\nJust columns with missing values:\n{users.loc[22:27, columns_with_missing]}")
    
    print('---------------------------------------------Ecode columns')
    # Create an empty dictionary ordinal_enc_dict
    ordinal_enc_dict = {}
    
    for col_name in users:
        # Create Ordinal encoder for col
        ordinal_enc_dict[col_name] = OrdinalEncoder()
        col = users[col_name]
        
        # Select non-null values of col
        col_not_null = col[col.notnull()]
        reshaped_vals = col_not_null.values.reshape(-1, 1)
        encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)
        
        # Store the values to non-null values of the column in users
        users.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
    print(users.loc[22:27, columns_with_missing])
    return(columns_with_missing, ordinal_enc_dict)
    
    
        
def KNN_imputation_of_categorical_values(columns_with_missing, ordinal_enc_dict):
    print("****************************************************")
    topic = "7. KNN imputation of categorical values"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------KNN Imputation')
    # Create KNN imputer
    KNN_imputer = KNN()    
    # Impute and round the users DataFrame
    users.iloc[:, :] = np.round(KNN_imputer.fit_transform(users))
    print(users.loc[22:27, columns_with_missing])

    print('---------------------------------------------Recode the columns')
    # Loop over the column names in users
    for col_name in users:
        # Reshape the data
        reshaped = users[col_name].values.reshape(-1, 1)
        # Perform inverse transform of the ordinally encoded columns
        users[col_name] = ordinal_enc_dict[col_name].inverse_transform(reshaped)
    print(users.loc[22:27, columns_with_missing])
    
    
    
def Evaluation_of_different_imputation_techniques():
    print("****************************************************")
    topic = "8. Evaluation of different imputation techniques"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Applying different imputations models')
    columns_with_missing, imputations_dict = Apply_different_imputations_models(diabetes)
    for imputations in imputations_dict:
        print(f"{imputations}:\n{imputations_dict[imputations][columns_with_missing].tail()}\n")
    
    print('---------------------------------------------Visualizing imputations')
    compare_imputations(columns_with_missing, imputations_dict, topic, ncols=2)
    
    print('---------------------------------------------Evaluating imputations')
    df_rsquared, df_rsquared_adj, df_coef, lm_dict, imputations_notnulls = Evaluate_imputation_model(imputations_dict)
    
    print(f"The model of the complete data (without nulls):\n{lm_dict['original data'].summary()}\n")
    print(f"{df_rsquared}\n\n\n{df_rsquared_adj}\n\n\n{df_coef}")
    
    print('---------------------------------------------Selecting the best model')
    # Get the R_squared_adj from the imputation model only
    r_squares_adj = dict(zip(df_rsquared_adj.columns[1:],df_rsquared_adj.values.flatten()[1:]))
    
    # Select best R-squared
    best_imputation = max(r_squares_adj, key=r_squares_adj.get)
    print("The best imputation technique is: ", best_imputation)
    
    print('---------------------------------------------Compare the density plot of the imputations')
    Visualize_KDE_from_imputations_model(columns_with_missing, imputations_notnulls, topic)
    return imputations_notnulls, lm_dict
    
    
    
def Analyze_the_summary_of_linear_model(diabetes_cc):
    """diabetes_cc is the diabetes withou nulls"""
    print("****************************************************")
    topic = "9. Analyze the summary of linear model"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Setting and printing the linear regression model')
    # Add constant to X and set X & y values to fit linear model
    X = sm.add_constant(diabetes_cc.iloc[:, :-1])
    y = diabetes_cc['Class']
    lm = sm.OLS(y, X).fit()
    
    # Print summary of lm
    print('\nSummary: ', lm.summary())
    
    # Print R squared score of lm
    print('\nAdjusted R-squared score: ', lm.rsquared_adj)
    
    # Print the params of lm
    print(f'\nCoefficcients:\n{lm.params}')
    
    
        
def Comparing_Rsquared_and_coefficients(lm, lm_mean, lm_KNN, lm_MICE):
    print("****************************************************")
    topic = "10. Comparing R-squared and coefficients"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Making the evaluation model')
    # Store the Adj. R-squared scores of the linear models
    r_squared = pd.DataFrame({'Complete Case': lm.rsquared_adj, 
                              'Mean Imputation': lm_mean.rsquared_adj, 
                              'KNN Imputation': lm_KNN.rsquared_adj, 
                              'MICE Imputation': lm_MICE.rsquared_adj}, 
                             index=['Adj. R-squared'])    
    print(r_squared)
        
    # Store the coefficients of the linear models
    coeff = pd.DataFrame({'Complete Case': lm.params, 
                          'Mean Imputation': lm_mean.params, 
                          'KNN Imputation': lm_KNN.params, 
                          'MICE Imputation': lm_MICE.params})    
    print(coeff)
        
    print('---------------------------------------------Selecting the best model')
    r_squares = {'Mean Imputation': lm_mean.rsquared_adj, 
                 'KNN Imputation': lm_KNN.rsquared_adj, 
                 'MICE Imputation': lm_MICE.rsquared_adj}
    # Select best R-squared
    best_imputation = max(r_squares, key=r_squares.get)
    print("The best imputation technique is: ", best_imputation)

    
        
def Comparing_density_plots(diabetes_cc, diabetes_mean_imputed, diabetes_knn_imputed, diabetes_mice_imputed):
    print("****************************************************")
    topic = "11. Comparing density plots"; print("** %s" % topic)
    print("****************************************************")
    column_name = 'Skin_Fold'
    
    print(f'---------------------------------------------Visualizing the kde for {column_name}')
    # Plot graphs of imputed DataFrames and the complete case
    fig, ax = plt.subplots(figsize=(12.1, 5.9))
    diabetes_cc[column_name].plot(kind='kde', c='red', linewidth=3, ax=ax)
    diabetes_mean_imputed[column_name].plot(kind='kde', ax=ax)
    diabetes_knn_imputed[column_name].plot(kind='kde', ax=ax)
    diabetes_mice_imputed[column_name].plot(kind='kde', ax=ax)
    
    # Create labels for the four DataFrames
    labels = ['Baseline (Complete Case)', 'Mean Imputation', 'KNN Imputation', 'MICE Imputation']
    ax.legend(labels)
    
    # Set the x-label as Skin Fold
    ax.set_xlabel(' ')
    ax.set_title(f"Column: '{column_name.upper()}'", **title_param)
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.7); #To set the margins 
    plt.show()
    
    
        
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Imputing_using_fancyimpute()
    #KNN_imputation()
    #MICE_imputation()
    
    Imputing_categorical_values()
    Ordinal_encoding_of_a_categorical_column(users)
    columns_with_missing, ordinal_enc_dict = Ordinal_encoding_of_a_DataFrame()
    KNN_imputation_of_categorical_values(columns_with_missing, ordinal_enc_dict)
    
    imputations_notnulls, lm_dict = Evaluation_of_different_imputation_techniques()
    Analyze_the_summary_of_linear_model(imputations_notnulls['original data']) #Sending the original Diabetes dataset without nulls
    Comparing_Rsquared_and_coefficients(lm_dict['original data'], lm_dict['mean imputer'], lm_dict['KNN imputer'], lm_dict['MICE imputer'])
    Comparing_density_plots(imputations_notnulls['original data'], imputations_notnulls['mean imputer'], imputations_notnulls['KNN imputer'], imputations_notnulls['MICE imputer'])
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    np.set_printoptions(formatter = {'float': None}) #Return to default
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    pd.set_option("display.max_columns", None)
