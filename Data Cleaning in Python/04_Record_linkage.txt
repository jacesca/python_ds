# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:11:28 2020

@author: jacesca@gmail.com
Chapter 4: Record linkage
    Record linkage is a powerful technique used to merge multiple datasets together, 
    used when values have typos or different spellings. In this chapter, you'll learn 
    how to link records by calculating the similarity between strings—you’ll then use 
    your new skills to join two restaurant review datasets into one clean master 
    dataset.
Source: https://learn.datacamp.com/courses/data-cleaning-in-python
Help:
    https://recordlinkage.readthedocs.io/en/latest/ref-datasets.html
    https://recordlinkage.readthedocs.io/en/latest/notebooks/link_two_dataframes.html
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import recordlinkage

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

###############################################################################
## Preparing the environment
###############################################################################
    
# Global variables
SEED = 42
np.random.seed(SEED) 

# Read the data
#from recordlinkage.datasets import load_febrl4
#census_A, census_B = load_febrl4()
#census_A.to_csv("census_A.csv", index=False)
#census_B.to_csv("census_B.csv", index=False)

restaurants_mismatch = pd.read_csv('restaurants_mismatch.csv', sep=';')
census_A = pd.read_csv('census_A.csv', nrows=500)
census_B = pd.read_csv('census_B.csv', nrows=500)
restaurants_old = pd.read_csv('restaurants_L2.csv')
restaurants_new = pd.read_csv('restaurants_L2_dirty.csv')


###############################################################################
## Main part of the code
###############################################################################
def Comparing_strings(seed=SEED):
    print("****************************************************")
    topic = "1. Comparing strings"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------SIMPLE STRING COMPARISON')
    # Compare reeding vs reading
    string1 = 'Reeding'
    string2 = 'Reading'
    wratio = fuzz.WRatio(string1, string2)
    print(f'The wratio from "{string1}" to "{string2}" is {wratio}')
    
    print('\n-------------PARTIAL STRINGS AND DIFFERENT ORDERINGS')
    # Partial string comparison
    string1 = ['Houston Rockets', 'Houston Rockets vs Los Angeles Lakers']
    string2 = ['Rockets', 'Lakers vs Rockets']
    
    for s1, s2 in zip(string1, string2):
        wratio = fuzz.WRatio(s1, s2)
        print(f'The wratio from "{s1}" to "{s2}" is {wratio}')
    
    
    print('\n-----------------------------COMPARTISON WITH ARRAYS')
    # Define string and array of possible matches
    string = "Houston Rockets vs Los Angeles Lakers"
    choices = pd.Series(['Rockets vs Lakers', 'Lakers vs Rockets','Houson vs Los Angeles', 'Heat vs Bulls'])
    wratio = process.extract(string, choices, limit = 2)
    print(f'The wratio of the comparison of the string "{string}" wiht the array: \n{choices}\n is \n{wratio}')
    
    
    print('\n----------COLLAPSING CATEGORIES WITH STRING MATCHING')
    survey = pd.DataFrame({'state'      : ['California', 'Cali', 'Calefornia 1','Calefornie','Californie','Calfornia','Calefernia','New York','New York City'], 
                           'move_scores': [1,1,1,3,0,2,0,2,2]})    
    categories = ['California', 'New York']
    print(f'Dataset:\n{survey} \n\nCategories: {categories}\n\n')
    
    # For each correct category
    for state in categories:
        # Find potential matches in states with typoes
        matches = process.extract(state, survey['state'], limit = survey.shape[0])
        print(f"Matches found for '{state}': \n{matches}")
        
        # For each potential match match
        for potential_match in matches:
            # If high similarity score
            if potential_match[1] >= 80:
                # Replace typo with correct category
                survey.loc[survey['state'] == potential_match[0], 'state'] = state
    
    print(f'\n\nCollapsed Dataset:\n{survey}')
    
    
    
def The_cutoff_point(seed=SEED):
    print("\n\n****************************************************")
    topic = "3. The cutoff point"; print("** %s" % topic)
    print("****************************************************")
    
    print('-------------------------------------------EXPLORING')
    print(restaurants_mismatch.head())

    print('--------------FINDING UNIQUE VALUES OF CUISSINE_TYPE')
    # Store the unique values of cuisine_type in unique_types
    unique_types = restaurants_mismatch.cuisine_type.unique()
    print(f"Unique types found in the dataset: \n{unique_types}")
    
    print('--------------------------------FINDING SIMILARITIES')
    categories = ['asian', 'american', 'italian']
    unique_len = len(unique_types)
    for catego in categories:
        similarity = process.extract(catego, unique_types, limit=unique_len)
        print(f"The similarity of '{catego}' to all values of unique types: \n{similarity}\n")
    
    
            
def Remapping_categories_II(seed=SEED):
    print("\n\n****************************************************")
    topic = "4. Remapping categories II"; print("** %s" % topic)
    print("****************************************************")
    
    print('-------------------------------------------EXPLORING')
    print(restaurants_mismatch.head())

    print('---------------------------CORRECT CATEGORIES TO MAP')
    categories = ['asian', 'american', 'italian']
    print(f"{categories}")
    
    print('--------------FINDING UNIQUE VALUES OF CUISSINE_TYPE')
    print(restaurants_mismatch['cuisine_type'].unique())
    
    print('------------------------------REMAPING CUISSINE TYPE')
    restaurant_len = restaurants_mismatch.shape[0]
    # For each correct cuisine_type in categories
    for cuisine in categories:
        # Find matches in cuisine_type of restaurants
        matches = process.extract(cuisine, restaurants_mismatch['cuisine_type'], limit = restaurant_len)
        
        # For each possible_match with similarity score >= 80
        for possible_match in matches:
            if possible_match[1] >= 80:
                # Find matching cuisine type
                matching_cuisine = restaurants_mismatch['cuisine_type'] == possible_match[0]
                restaurants_mismatch.loc[matching_cuisine, 'cuisine_type'] = cuisine
                
    # Print unique values to confirm mapping
    print(restaurants_mismatch['cuisine_type'].unique())
    
    print('----------------------------------------FINAL RESULT')
    print(restaurants_mismatch.head())
    
    
    
def Generating_pairs(seed=SEED):
    print("\n\n****************************************************")
    topic = "5. Generating pairs"; print("** %s" % topic)
    print("****************************************************")
    
    print('-------------------------------------------EXPLORING')
    print(f"*** Census A ***\nColumns: {census_A.columns}")
    print(f"Head:\n {census_A.head()}\n\n")
    print(f"*** Census B ***\nColumns: {census_B.columns}")
    print(f"Head:\n {census_B.head()}\n\n")
    
    print('------------------------------------GENERATING PAIRS')
    # Create indexing object
    indexer = recordlinkage.Index()
    # Generate pairs blocked on state
    indexer.block('state',)
    pairs = indexer.index(census_A, census_B)
    print(f"Just exploring pairs maded: \n{pairs}\n")
    
    print('----------------------------COMPARING THE DATAFRAMES')
    # Create a Compare object
    compare_cl = recordlinkage.Compare()
    # Find exact matches for pairs of date_of_birth and state
    compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
    compare_cl.exact('state', 'state', label='state')
    # Find similar matches for pairs of surname and address_1 using string similarity
    compare_cl.string('surname', 'surname', threshold=0.85, label='surname')
    compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')
    # Find matches
    potential_matches = compare_cl.compute(pairs, census_A, census_B)
    print(f"Just exploring matches ({potential_matches.shape[0]} rows): \n{potential_matches.head()}\n")
        
    print('------------------FINDING THE PAIRS WE PROBABLY WANT')
    definite_matches = potential_matches[potential_matches.sum(axis = 1) >= 2]
    print(f"The pairs requiered ({definite_matches.shape[0]} rows): \n{definite_matches.head()}\n")
    return potential_matches
    
    
        
def Pairs_of_restaurants(seed=SEED):
    print("\n\n****************************************************")
    topic = "7. Pairs of restaurants"; print("** %s" % topic)
    print("****************************************************")
    
    print('-------------------------------------------EXPLORING')
    print(f"*** Restaurants (old list) ***\nColumns: {restaurants_old.columns}")
    print(f"Head:\n {restaurants_old.head()}\n\n")
    print(f"*** Restaurants (new list) ***\nColumns: {restaurants_new.columns}")
    print(f"Head:\n {restaurants_new.head()}\n\n")
    
    print('------------------------------------GENERATING PAIRS')
    # Create indexing object
    indexer = recordlinkage.Index()
    # Generate pairs blocked on state
    indexer.block('type')
    # Generate pairs
    pairs = indexer.index(restaurants_old, restaurants_new)
    print(f"Just exploring pairs maded (size: {pairs.shape[0]}): \n{pairs}\n")
    return pairs
     
    
def Similar_restaurants(pairs, seed=SEED):
    print("\n\n****************************************************")
    topic = "8. Similar restaurants"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------COMPARING THE DATAFRAMES')
    # Create a comparison object
    comp_cl = recordlinkage.Compare()
    # Find exact matches on city, cuisine_types 
    comp_cl.exact('city', 'city', label = 'city')
    comp_cl.exact('type', 'type', label = 'cuisine_type')
    # Find similar matches of rest_name
    comp_cl.string('name', 'name', label='rest_name', threshold = .8) 
    # Get potential matches and print
    potential_matches = comp_cl.compute(pairs, restaurants_old, restaurants_new)
    print(f"Just exploring matches ({potential_matches.shape[0]} rows): \n{potential_matches.head()}\n")
        
    print('---------------------------FINDING THE PAIRS WE WANT')
    definite_matches = potential_matches[potential_matches.sum(axis = 1) >= 3]
    print(f"The pairs requiered ({definite_matches.shape[0]} rows): \n{definite_matches.head()}\n")
    return definite_matches
    
    
    
def Linking_DataFrames(potential_matches, seed=SEED):
    print("\n\n****************************************************")
    topic = "9. Linking DataFrames"; print("** %s" % topic)
    print("****************************************************")
    
    print('-------------------------------------------EXPLORING')
    print(f"*** Census A ***\nColumns: {census_A.columns}")
    print(f"Head ({census_A.shape[0]} rows):\n {census_A.head()}\n\n")
    print(f"*** Census B ***\nColumns: {census_B.columns}")
    print(f"Head ({census_B.shape[0]} rows):\n {census_B.head()}\n\n")
    
    print('---------------------------FINDING THE PAIRS WE WANT')
    print(f"Potential matches: {potential_matches.shape[0]} rows.")
    matches = potential_matches[potential_matches.sum(axis = 1) >= 3]
    print(f"The pairs requiered ({matches.shape[0]} rows): \n{matches.head()}\n")
    
    print('---------------------------------------GET THE INDEX')
    print(f"Index of mathes required ({matches.shape[0]}): \n{matches.index}")
    # Get indices from census_B only
    duplicate_rows = matches.index.get_level_values(1)
    print(f"Duplicates only ({duplicate_rows.shape[0]}): {duplicate_rows}")

    print('----------------------------------LINKING DATAFRAMES')
    # Finding duplicates in census_B
    census_B_duplicates = census_B[census_B.index.isin(duplicate_rows)]
    print(f"Exploring duplicates ({census_B_duplicates.shape[0]} rows):\n {census_B_duplicates}\n\n")
    
    # Finding new rows in census_B
    census_B_new = census_B[~census_B.index.isin(duplicate_rows)]
    # Link the DataFrames!
    full_census = census_A.append(census_B_new)
    print(f"Final census set ({full_census.shape[0]} rows):\n {full_census}\n\n")
    
    
    
def Linking_them_together(definite_matches, seed=SEED):
    print("\n\n****************************************************")
    topic = "11. Linking them together!"; print("** %s" % topic)
    print("****************************************************")
    
    print('-------------------------------------------EXPLORING')
    print(f"{restaurants_old.shape[0]} rows in restaurant_old.")
    print(f"{restaurants_new.shape[0]} rows in restaurant_new.")
    
    print('---------------------------------------GET THE INDEX')
    # Get values of second column index of matches
    matching_indices = definite_matches.index.get_level_values(1)
    print(f"Matches found: {matching_indices.shape[0]}\n")
    
    print('--------------------EXPLORING DUPLICATES RESTAURANTS')
    duplicates_rows_in_restaurants_new = restaurants_new.index.isin(matching_indices)
    duplicate_restaurants_new = restaurants_new[duplicates_rows_in_restaurants_new].sort_values(by=['type', 'phone'])
    print(f"{duplicate_restaurants_new.shape[0]} rows found in restaurants_new:\n {duplicate_restaurants_new}\n")
    
    duplicate_restaurants_old = restaurants_old[restaurants_old.index.isin(definite_matches.index.get_level_values(0))].sort_values(by=['type', 'phone'])
    print(f"This correspond to the next rows in restaurant_old:\n {duplicate_restaurants_old}\n\n")

    print('--------------------------NON DUPLICATES RESTAURANTS')
    # Subset restaurants_new based on non-duplicate values
    non_dup = restaurants_new[~duplicates_rows_in_restaurants_new]
    print(f"{non_dup.shape[0]} rows non duplicate found.\n")
    
    print('----------------------------------LINKING DATAFRAMES')
    # Link the DataFrames!
    full_restaurants = restaurants_old.append(non_dup)
    print(f"Final restaurants set ({full_restaurants.shape[0]} rows):\n {full_restaurants}\n\n")
    
    
    
def main(seed=SEED):
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Comparing_strings()
    The_cutoff_point()
    Remapping_categories_II()
    census_potential_matches = Generating_pairs()
    restaurant_pairs = Pairs_of_restaurants()
    restaurant_definite_matches = Similar_restaurants(restaurant_pairs)
    Linking_DataFrames(census_potential_matches)
    Linking_them_together(restaurant_definite_matches)
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()