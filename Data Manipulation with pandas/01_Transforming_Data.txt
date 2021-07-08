# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:37:38 2020

@author: jacesca@gmail.com
Chapter 1: Transforming Data
    Let’s master the pandas basics. Learn how to inspect DataFrames and perform 
    fundamental manipulations, including sorting rows, subsetting, and adding new 
    columns.
Source: https://learn.datacamp.com/courses/data-manipulation-with-pandas
"""
###############################################################################
## Importing libraries
###############################################################################
#import numpy as np
import pandas as pd

###############################################################################
## Preparing the environment
###############################################################################
    
# Global variables
#SEED = 42
#np.random.seed(SEED) 

# Read the data
homelessness = pd.read_pickle("homeless_data.pkl")


###############################################################################
## Main part of the code
###############################################################################
def Inspecting_a_DataFrame():
    print("****************************************************")
    topic = "2. Inspecting a DataFrame"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------------------HEAD')
    # Print the head of the homelessness data
    print(homelessness.head(),"\n")
    
    print('-----------------------------------------INFORMATION')
    # Print information about homelessness
    print(homelessness.info(),"\n")
    
    print('--------------------------NUMBER OF ROWS AND COLUMNS')
    # Print the shape of homelessness
    print(homelessness.shape,"\n")
    
    print('----------------------------------SUMMARY STATISTICS')
    # Print a description of homelessness
    print(homelessness.describe())
    
    
def Parts_of_a_DataFrame():
    print("\n\n****************************************************")
    topic = "3. Parts of a DataFrame"; print("** %s" % topic)
    print("****************************************************")
    
    print('-----------------------------------------DATA VALUES')
    # Print the values of homelessness
    print(homelessness.values,"\n")
    
    print('---------------------------------------------COLUMNS')
    # Print the column index of homelessness
    print(homelessness.columns,"\n")
    
    print('------------------------------------------------ROWS')
    # Print the row index of homelessness
    print(homelessness.index)
    
    
    
def Sorting_rows():
    print("\n\n****************************************************")
    topic = "5. Sorting rows"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------ORDER BY ONE COLUMN: INDIVIDUALS')
    # Sort homelessness by individual
    homelessness_ind = homelessness.sort_values(by='individuals')
    # Print the top few rows
    print(homelessness_ind.head(),"\n")
    
    print('-----------ORDER BY ONE COLUMN: FAMILY_MEMBERS, DESC')
    # Sort homelessness by descending family members
    homelessness_fam = homelessness.sort_values(by='family_members', ascending=False)    
    # Print the top few rows
    print(homelessness_fam.head(),"\n")

    print('-----------ORDER BY: REGION ASC, FAMILY_MEMBERS DESC')
    # Sort homelessness by region, then descending family members
    homelessness_reg_fam = homelessness.sort_values(by=['region', 'family_members'], ascending=[True, False])
    # Print the top few rows
    print(homelessness_reg_fam.head())
    
    
    
def Subsetting_columns():
    print("\n\n****************************************************")
    topic = "6. Subsetting columns"; print("** %s" % topic)
    print("****************************************************")
    
    print('-------------------SELECTING ONE COLUMN: INDIVIDUALS')
    # Select the individuals column
    individuals = homelessness[['individuals']]
    # Print the head of the result
    print(individuals.head(),"\n")
    
    print('-----SELECTING TWO COLUMNS: STATE AND FAMILY_MEMBERS')
    # Select the state and family_members columns
    state_fam = homelessness[['state', 'family_members']]
    # Print the head of the result
    print(state_fam.head(),"\n")
    
    print('--------SELECTING TWO COLUMNS: INDIVIDUALS AND STATE')
    # Select only the individuals and state columns, in that order
    ind_state = homelessness[['individuals', 'state']]
    # Print the head of the result
    print(ind_state.head())



def Subsetting_rows():
    print("\n\n****************************************************")
    topic = "7. Subsetting rows"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------FILTER BY INDIVIDUALS>10K')
    # Filter for rows where individuals is greater than 10000
    ind_gt_10k = homelessness[homelessness.individuals > 1e+4]
    # See the result
    print(ind_gt_10k, '\n')
    
    print('---------------------------FILTER BY MOUNTAIN REGION')
    # Filter for rows where region is Mountain
    mountain_reg = homelessness[homelessness.region=='Mountain']
    # See the result
    print(mountain_reg, '\n')
    
    print('------FILTER BY FAMILY_MEMBERS<1K AND PACIFIC REGION')
    # Filter for rows where family_members is less than 1000 
    # and region is Pacific
    fam_lt_1k_pac = homelessness[(homelessness.family_members<1000) & (homelessness.region=='Pacific')]
    # See the result
    print(fam_lt_1k_pac)
    
    
def Subsetting_rows_by_categorical_variables():
    print("\n\n****************************************************")
    topic = "8. Subsetting rows by categorical variables"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------FILTERING CATEGORICALS: REGION')
    # Subset for rows in South Atlantic or Mid-Atlantic regions
    south_mid_atlantic = homelessness[homelessness.region.isin(['South Atlantic', 'Mid-Atlantic'])]
    # See the result
    print(south_mid_atlantic)
    
    print('----------------------FILTERING CATEGORICALS: STATES')
    # The Mojave Desert states
    canu = ["California", "Arizona", "Nevada", "Utah"]
    # Filter for rows in the Mojave Desert states
    mojave_homelessness = homelessness[homelessness.state.isin(canu)]
    # See the result
    print(mojave_homelessness)
    
    
    
def Adding_new_columns():
    print("\n\n****************************************************")
    topic = "10. Adding new columns"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------ADDING TOTAL AND P_INDIVIDUALS')
    # Add total col as sum of individuals and family_members
    homelessness['total'] = homelessness.individuals + homelessness.family_members
    # Add p_individuals col as proportion of individuals
    homelessness['p_individuals'] = homelessness.individuals / homelessness.state_pop
    # See the result
    print(homelessness.head())
    
    
    
def Combo_attack():
    print("\n\n****************************************************")
    topic = "11. Combo-attack!"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------CREATE NEW COLUMN: INDIV_PER_10K')
    # Create indiv_per_10k col as homeless individuals per 10k state pop
    homelessness["indiv_per_10k"] = 10000 * homelessness.individuals / homelessness.state_pop 
    
    print('-----------------------FILTERING: INDIV_PER_10K > 20')
    # Subset rows for indiv_per_10k greater than 20
    high_homelessness = homelessness[homelessness.indiv_per_10k > 20]
    
    print('-----------------------SORT VALUES: BY INDIV_PER_10K')
    # Sort high_homelessness by descending indiv_per_10k
    high_homelessness_srt = high_homelessness.sort_values(by='indiv_per_10k', ascending=False)
    
    print('-----------------------SETTING THE COLUMNS TO CHOOSE')
    # From high_homelessness_srt, select the state and indiv_per_10k cols
    result = high_homelessness_srt[['state', 'indiv_per_10k']]
    
    print('--------------------------------EXPLORING THE RESULT')
    # See the result
    print(result)
    
    
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Inspecting_a_DataFrame()
    Parts_of_a_DataFrame()
    Sorting_rows()
    Subsetting_columns()
    Subsetting_rows()
    Subsetting_rows_by_categorical_variables()
    Adding_new_columns()
    Combo_attack()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()