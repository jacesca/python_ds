# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:11:28 2020

@author: jacesca@gmail.com
Chapter 2: Text and categorical data problems
    Categorical and text data can often be some of the messiest parts of a dataset 
    due to their unstructured nature. In this chapter, you’ll learn how to fix 
    whitespace and capitalization inconsistencies in category labels, collapse 
    multiple categories into one, and reformat strings for consistency.
Source: https://learn.datacamp.com/courses/data-cleaning-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import datetime as dt
import numpy as np
import pandas as pd


###############################################################################
## Preparing the environment
###############################################################################
    
# Global variables
SEED = 42
np.random.seed(SEED) 

# Read the data
airlines = pd.read_csv('airlines_final.csv', index_col=0)
    
adult = pd.read_csv('adult.csv')
adult['income'] = np.round(np.random.random(size=len(adult))*870e+3, 2)

airlines_categories = pd.DataFrame({'cleanliness' : ['Clean', 'Average', 'Somewhat clean', 'Somewhat dirty', 'Dirty'], 
                                    'safety'      : ['Neutral', 'Very safe', 'Somewhat safe', 'Somewhat unsafe', 'Very unsafe'], 
                                    'satisfaction': ['Very satisfied', 'Neutral', 'Somewhat satsified', 'Somewhat unsatisfied', 'Very unsatisfied']})


###############################################################################
## Main part of the code
###############################################################################
def Membership_constraints(seed=SEED):
    print("****************************************************")
    topic = "1. Membership constraints"; print("** %s" % topic)
    
    # Prepare the data
    blood_categories = pd.DataFrame({'blood_type': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+']})
    study_data = pd.DataFrame({'name'      : ['Beth','Ignatius','Paul', 'Helen', 'Jennifer', 'Kennedy', 'Keith'],
                               'birthday'  : ['2019-10-20', '2020-07-08', '2019-08-12', '2019-03-17', '2019-12-17', '2020-04-27', '2019-04-19'],
                               'blood_type': ['B-', 'A-', 'O+', 'O-', 'Z+', 'A+', 'AB+']})
    
    print('-------------------------------------------EXPLORING')
    print("dataset: \n{}".format(study_data))
    print("valid blood categories: {}".format(blood_categories.blood_type.values))
    
    print('---------------------FINDING INCONSISTENT CATEGORIES')
    # Finding inconsistent categories
    inconsistent_categories = set(study_data['blood_type']).difference(blood_categories['blood_type'])
    print("inconsistent categories: {}".format(inconsistent_categories))
    
    # Get and print rows with inconsistent categories
    inconsistent_rows = study_data['blood_type'].isin(inconsistent_categories)
    inconsistent_data = study_data[inconsistent_rows]
    print("inconsistent data:\n{}".format(inconsistent_data))
    
    print('--------------------DROPPING INCONSISTENT CATEGORIES')
    # Drop inconsistent categories and get consistent data only
    consistent_data = study_data[~inconsistent_rows]
    print("consistent data:\n{}".format(consistent_data))
    
def Finding_consistency(seed=SEED):
    print("\n\n****************************************************")
    topic = "3. Finding consistency"; print("** %s" % topic)
    
    # Prepare data
    airlines = pd.read_csv('airlines_bad.csv')
    
    print('-------------------------------------------EXPLORING')
    # Print categories DataFrame
    print("valid categories: \n{}\n".format(airlines_categories))
    
    # Print unique values of survey columns in airlines
    print("Values found in airline dataset:")
    print('Cleanliness: ', airlines['cleanliness'].unique(), "\n")
    print('Safety: ', airlines.safety.unique(), "\n")
    print('Satisfaction: ', airlines.satisfaction.unique(), "\n")
    
    print('---------------------------FINDING INCONSISTING DATA')
    # Find the cleanliness category in airlines not in categories
    cat_clean = set(airlines.cleanliness).difference(airlines_categories.cleanliness)
    
    # Find rows with that category
    cat_clean_rows = airlines['cleanliness'].isin(cat_clean)
    
    # Print rows with inconsistent category
    print(airlines[cat_clean_rows])

    print('------------------------PRINTING THE CONSISTENT DATA')
    # Print rows with consistent categories only
    print(airlines[~cat_clean_rows])
    
        
        
        
def Categorical_variables(seed=SEED):
    print("\n\n****************************************************")
    topic = "4. Categorical variables"; print("** %s" % topic)
    
    print('-------------------------------------------EXPLORING')
    print(adult.columns,'\n')
    
    print('-----------------------------------VALUE CONSISTENCY')
    # Get marriage status column
    marriage_status = adult['marital_status']
    print("values in marital_status:\n{}\n".format(np.sort(marriage_status.unique())))
    print(marriage_status.value_counts(),'\n')
    
    # Get value counts on DataFrame
    print(adult.groupby('marital_status').count())
    
    # Capitalize
    adult['marital_status'] = adult['marital_status'].str.upper()
    print("capitalize: \n{}\n".format(np.sort(marriage_status.unique())))
    
    # Lowercase
    adult['marital_status'] = adult['marital_status'].str.lower()
    print("lowercase: \n{}\n".format(np.sort(marriage_status.unique())))
    
    # Trailing spaces:- strip all spaces
    adult['marital_status'] = adult['marital_status'].str.strip()
    print("strip space: \n{}\n".format(np.sort(marriage_status.unique())))
    
    print('-----------------COLLAPSING DATA INTO CATEGORIES (1)')
    # Using qcut()
    group_names = ['0-200K', '200K-500K', '500K+']
    adult['income_group'] = pd.qcut(adult['income'], 
                                    q = 3, labels = group_names)
    print(adult[['income', 'income_group']])

    print('-----------------COLLAPSING DATA INTO CATEGORIES (2)')
    print("workclass: \n{}\n".format(np.sort(adult.workclass.unique())))
    
    # Create mapping dictionary and replace
    mapping = {'Federal-gov':'Government', 'Local-gov':'Government', 'State-gov':'Government',
               'Private':'Private', 'Self-emp-inc':'Private', 'Self-emp-not-inc':'Private', 'Without-pay':'Private',
               'Never-worked':'Unemployed', '?':'Unemployed'}
    adult['workclass'] = adult['workclass'].replace(mapping)
    print("collapsed workclass: \n{}\n".format(np.sort(adult.workclass.unique())))
    
    
    
def Inconsistent_categories(seed=SEED):
    print("\n\n****************************************************")
    topic = "6. Inconsistent categories"; print("** %s" % topic)
    
    print('-------------------------------------------EXPLORING')
    # Print unique values of both columns
    print('dest_region: \n{}\n'.format(airlines['dest_region'].unique()))
    print('dest_size: \n{}\n'.format(airlines['dest_size'].unique()))
    
    print('------------------------------WORKING ON DEST_REGION')
    # Lower dest_region column and then replace "eur" with "europe"
    airlines['dest_region'] = airlines['dest_region'].str.lower()
    airlines['dest_region'] = airlines['dest_region'].replace({'eur':'europe'})
    print('dest_region: \n{}\n'.format(airlines['dest_region'].unique()))
    
    print('--------------------------------WORKING ON DEST_SIZE')
    # Remove white spaces from `dest_size`
    airlines['dest_size'] = airlines['dest_size'].str.strip()
    print('dest_size: \n{}\n'.format(airlines['dest_size'].unique()))
    


def Remapping_categories(seed=SEED):
    print("\n\n****************************************************")
    topic = "7. Remapping categories"; print("** %s" % topic)
    
    print('-------------------------------------------EXPLORING')
    # Print unique values of both columns
    print('applying qcut to wait_min: \n{}\n'.format(pd.qcut(airlines['wait_min'], q = 3).unique()))
    print('day: \n{}\n'.format(airlines['day'].unique()))
    
    print('---------------------------------WORKING ON WAIT_MIN')
    # Create ranges for categories
    label_ranges = [0, 60, 180, np.inf]
    label_names = ['short', 'medium', 'long']
    
    # Create wait_type column
    airlines['wait_type'] = pd.cut(airlines.wait_min, bins = label_ranges, 
                                   labels = label_names)
    print('wait_type: \n{}\n'.format(airlines['wait_type'].unique()))
    
    print('--------------------------------------WORKING ON day')
    # Create mappings and replace
    mappings = {'Monday':'weekday', 'Tuesday':'weekday', 'Wednesday': 'weekday', 
                'Thursday': 'weekday', 'Friday': 'weekday', 
                'Saturday': 'weekend', 'Sunday': 'weekend'}
    
    airlines['day_week'] = airlines['day'].replace(mappings)
    print('day_week: \n{}\n'.format(airlines['day_week'].unique()))
    
    
    
def Cleaning_text_data(seed=SEED):
    print("\n\n****************************************************")
    topic = "8. Cleaning text data"; print("** %s" % topic)
    
    # Prepare the data
    phones = pd.DataFrame({'full_name'    : ['Noelani A. Gray', 'Myles Z. Gomez', 'Gil B. Silva', 'Prescott D. Hardin', 'Benedict G. Valdez', 'Reece M. Andrews', 'Hayfa E. Keith', 'Hedley I. Logan', 'Jack W. Carrillo', 'Lionel M. Davis'],
                           'phone_number' : ['001-702-397-5143', '+1-329-485-0540', '4138', '+1-297-996-4904', '001-969-820-3536', '001-195-492-2338', '001-536-175-8444', '+1-681-552-1823', '001-910-323-5265', '001-143-119-9210']})
    print('-------------------------------------------EXPLORING')
    print(phones.head(3),'\n')
    
    print('-----------REPLACE DASH USING REGULAR EXPRESSION (1)')
    # Replace letters with nothing
    phones["phone_formatted"] = phones.phone_number.str.replace(r'\D+', '')
    print(phones.head(3),'\n')
    
    print('------------------------------------REPLACE DASH (1)')
    # Replace "-" with nothing
    phones["phone_formatted"] = phones.phone_number.str.replace("-", "")
    print(phones.head(3),'\n')
    
    print('---------------------------------------REPLACE + (2)')
    # Replace "-" with nothing
    phones["phone_formatted"] = phones.phone_formatted.str.replace("+", "00")
    print(phones.head(3),'\n')
    
    print('---------------REPLACE PHONE NUMBER LESS THAN 10 (3)')
    # Replace phone numbers with lower than 10 digits to NaN
    digits = phones['phone_formatted'].str.len()
    print("phone len: \n{}\n".format(digits.head(3)))
    phones.loc[digits < 10, "phone_formatted"] = np.nan
    print(phones.head(3),'\n')
    
    print('---------------MAKING SOME ASSERTION TO VALIDATE (4)')
    # Assert minmum phone number length is 10
    try: 
        sanity_check = phones['phone_formatted'].str.len()
        assert sanity_check.min() >= 10
        print("len phone number ok!...\n")
    except: print("Error: phone number len lower than 10!...\n")
    
    # Assert all numbers do not have "+" or "-"
    try:
        phone_corrected = phones['phone_formatted'].str.contains("\+|\-", regex=True, na=False)
        print("phones with - or +: \n{}\n".format(phone_corrected.head(3)))
        assert  phone_corrected.any() == False
        print("Pones without + or - ok!...")
    except: print("Error: phone number with + or -!...")
    
    
    
def Removing_titles_and_taking_names(seed=SEED):
    print("\n\n****************************************************")
    topic = "9. Removing titles and taking names"; print("** %s" % topic)
    
    # Prepare the data
    airlines = pd.read_csv('airlines_names.csv', sep=';', index_col=0)
    
    print('-------------------------------------------EXPLORING')
    print(airlines.full_name,'\n')
    
    print('----------------------------------REPLACE HONORIGICS')
    # Replace "Dr.|Mr.|Miss|Ms." with empty string ""
    airlines['full_name'] = airlines['full_name'].str.replace(r'Dr.\s|Mr.\s|Miss\s|Ms.\s', '')
    print(airlines.full_name,'\n')
    
    print('---------------------SEPARATE IN FIRST AND LAST NAME')
    # Replace "Dr.|Mr.|Miss|Ms." with empty string ""
    airlines['separate_name'] = airlines['full_name'].str.split('\s')
    airlines['first_name'] = airlines["separate_name"].str.get(0)
    airlines['last_name'] = airlines["separate_name"].str.get(1)
    
    print(airlines[['full_name', 'separate_name', 'first_name', 'last_name']],'\n')
    
    print('-------------------MAKING SOME ASSERTION TO VALIDATE')
    # Assert that full_name has no honorifics
    try:
        assert airlines['full_name'].str.contains('Ms.|Mr.|Miss|Dr.').any() == False
        print("names without honorifics ok!...")
    except: print("Error: names have honorifics yet!...")
        
    
    
def Treating_duplicates(seed=SEED):
    print("\n\n****************************************************")
    topic = "10. Treating duplicates"; print("** %s" % topic)
    
    # Prepare the data
    airlines = pd.read_csv('airlines_survey.csv', sep=';')
    
    print('-------------------------------------------EXPLORING')
    print(airlines.survey_response,'\n')
    
    print('--------------------IDENTIFYING RELEVANT INFORMATION')
    # Store length of each row in survey_response column
    resp_length = airlines.survey_response.str.len()
    print('comment len: \n{}\n'.format(resp_length))
    
    # Find rows in airlines where resp_length > 40
    airlines_survey = airlines[resp_length > 40]
    print('comment bigger than 40 chars: \n{}\n'.format(airlines_survey))
    
    print('-------------------MAKING SOME ASSERTION TO VALIDATE')
    # Assert minimum survey_response length is > 40
    try:
        assert airlines_survey.survey_response.str.len().min() > 40
        print("all identified survey bigger than 40 chars ok!...")
    except: print("Error: there are comment with less than 40 chars!...")

    
    

def main(seed=SEED):
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Membership_constraints()
    Finding_consistency()
    Categorical_variables()
    Inconsistent_categories()
    Remapping_categories()
    Remapping_categories()
    Cleaning_text_data()
    Removing_titles_and_taking_names()
    Treating_duplicates()
        
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()