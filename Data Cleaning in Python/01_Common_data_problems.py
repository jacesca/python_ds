# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:11:28 2020

@author: jacesca@gmail.com
Chapter 1: Common data problems
    In this chapter, you'll learn how to overcome some of the most common dirty 
    data problems. You'll convert data types, apply range constraints to remove 
    future data points, and remove duplicated data points to avoid 
    double-counting.
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

# Global utilities functions
def rand_datetime(start=dt.date.today().replace(day=1, month=1), end=dt.date.today(), size=1):
    init_date = start.toordinal()
    end_date = end.toordinal()
    rand_dates = np.random.randint(low=init_date, high=end_date, size=size)
    return [dt.date.fromordinal(item) for item in rand_dates]

# Read the data
tips = pd.read_csv('tips.csv')
mlb_players = pd.read_csv('mlb_players.csv')

ride_sharing = pd.read_csv('ride_sharing_new.csv', index_col=0)
ride_duplicated = pd.read_csv('ride_duplicated.csv', index_col=0, sep=';')

airlines = pd.read_csv('airlines_final.csv')

###############################################################################
## Main part of the code
###############################################################################
def Data_type_constraints(seed=SEED):
    print("****************************************************")
    topic = "1. Data type constraints"; print("** %s" % topic)
    
    # Applying str.strip(char)
    print('-----------------------------------APPYING STR.STRIP')
    my_str_test = '42$'; print("{} is transformed in {}".format(my_str_test, my_str_test.strip('$')))
    my_str_test = '$42'; print("{} is transformed in {}".format(my_str_test, my_str_test.strip('$')))
    my_str_test = '4$2'; print("{} is transformed in {}".format(my_str_test, my_str_test.strip('$')))

    
    # Transformint to a str
    print('---------------------------------TRANSFORMING TO STR')
    tips['size'] = tips['size'].apply(lambda x: str(x) + '°')
    #print(tips.head(1))
    #print(tips.info())
    #print(tips['size'].sum())
    
    # Transfprming to integer
    print('---------------------------------TRANSFORMING TO INT')
    tips['size'] = tips['size'].str.strip('°')
    tips['size'] = tips['size'].astype(int)
    #print(tips.head(1))
    #print(tips.info())
    #print(tips['size'].sum())
    
    # Being sure that the column is integer
    print('--------------------------------VERIFY IT IS INTEGER')
    try   : 
        assert tips['size'].dtype == 'int'
        print("It's integer!")
    except: print("Error: Not integer!...")
    
    print('-------------------------CREATING A NUMERICAL COLUMN')
    # Creating a numerical column
    tips['gender'] = tips.sex.map({'Female': 0, 'Male': 1})
    #print(tips.head(2))
    #print(tips.info())
    #print(tips.gender.describe())
    #print(tips.sex.describe())
    
    # Transforming to a categorical
    print('-------------------------TRANSFORMING TO CATEGORICAL')
    tips['gender'] = tips.gender.astype('category')
    tips['sex'] = tips.sex.astype('category')
    
    print('-------------------------------------------EXPLORING')
    #print(tips.info())
    #print(tips.gender.describe())
    #print(tips.sex.describe())
    print(tips.head(2))
    
    
    
def Numeric_data_or(seed=SEED):
    print("\n\n****************************************************")
    topic = "3. Numeric data or ... ?"; print("** %s" % topic)
    
    print('-------------------------TRANSFORMING TO CATEGORICAL')
    ride_sharing['user_type_cat'] = ride_sharing['user_type'].astype('category') # Convert user_type from integer to category
    
    # Write an assert statement confirming the change
    try   : 
        assert ride_sharing['user_type_cat'].dtype == 'category'
        print("It's category!")
    except: print("Error: Not category!...")
    
    # Print new summary statistics 
    #print(ride_sharing['user_type_cat'].describe())
    #print(ride_sharing.info()) # Print the information of ride_sharing
    
    print('-------------------------------------------EXPLORING')
    #print(ride_sharing.info()) # Print the information of ride_sharing
    #print(ride_sharing['user_type'].describe()) # Print summary statistics of user_type column
    print(ride_sharing.head(2)) # Print the head of ride_sharing
    
    
    
def Summing_strings_and_concatenating_numbers(seed=SEED):
    print("\n\n****************************************************")
    topic = "4. Summing strings and concatenating numbers"; print("** %s" % topic)
    
    # Transfprming to integer
    print('---------------------------------TRANSFORMING TO INT')
    ride_sharing['duration_trim'] = ride_sharing['duration'].str.strip('minutes') # Strip duration of minutes (without spaces)
    ride_sharing['duration_time'] = ride_sharing['duration_trim'].astype(int) # Convert duration to integer
    
    # Write an assert statement making sure of conversion
    try   : assert ride_sharing['duration_time'].dtype == 'int'
    except: print("Error: Not categorical...")
    
    print('-------------------------------------------EXPLORING')
    # Print formed columns and calculate average ride duration 
    print('duration time mean: ', ride_sharing['duration_time'].mean())
    #print(ride_sharing.info()) # Print the information of ride_sharing
    print(ride_sharing[['duration','duration_trim','duration_time']].head(2))
        
        
        
def Data_range_constraints(seed=SEED):
    print("\n\n****************************************************")
    topic = "5. Data range constraints"; print("** %s" % topic)
    
    # Transfprming to integer
    print('-----------------------------------------------TODAY')
    print(dt.date.today())
    
    print('----------------------------TRANSFORMING TO DATETIME')
    airlines['dept_time'] = pd.to_datetime(airlines.dept_time, yearfirst=True, format="%Y-%m-%d")
    #print(airlines.info()) 
    
    # Assert that conversion happened
    try   : 
        airlines.dept_time.dtype == 'datetime64[ns]'
        print("Datetime column, ok!...")
    except: print("Error: Not datetime!")
    
    print('mean: ', airlines.dept_time.mean())
    print(airlines[airlines.dept_time >= airlines.dept_time.mean()][['dept_time']].size, ' rows greater than mean date...')
    
    print('-------------------------------------------EXPLORING')
    #print(airlines.info()) 
    print(airlines[['dept_time']].head(2)) 
    
    
    
def Tire_size_constraints(seed=SEED):
    print("\n\n****************************************************")
    topic = "6. Tire size constraints"; print("** %s" % topic)
    
    #Preparing the data
    np.random.seed(seed)
    tire_sizes = ['26', '27', '29']
    ride_sharing["tire_sizes"] = np.random.choice(tire_sizes, size=len(ride_sharing))
    
    print('---------------------------------TRANSFORMING TO INT')
    # Convert tire_sizes to integer
    ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('int')
    
    print('---------------------------------------SET THE RANGE')
    # Set all values above 27 to 27
    ride_sharing.loc[ride_sharing['tire_sizes'] > 27, 'tire_sizes'] = 27
    
    # Assert that maximum value of tire_sizes is 27
    try   : assert ride_sharing['tire_sizes'].max() == 27
    except: print("Error: Bigger than 27...")
    print(ride_sharing.tire_sizes.unique())
    
    print('-------------------------TRANSFORMING TO CATEGORICAL')
    # Reconvert tire_sizes back to categorical
    ride_sharing['tire_sizes'] = ride_sharing.tire_sizes.astype('category')
    
    print('-------------------------------------------EXPLORING')
    # Print tire size description
    #print(ride_sharing['tire_sizes'].describe())
    #print(ride_sharing.info()) 
    print(ride_sharing.head(2)) 
    


def Back_to_the_future(seed=SEED):
    print("\n\n****************************************************")
    topic = "7. Back to the future"; print("** %s" % topic)
    
    #Preparing the data
    np.random.seed(seed)
    ride_sharing['ride_date'] = rand_datetime(end=dt.date.today().replace(day=31, month=5), size=len(ride_sharing))
    
    print('----------------------------TRANSFORMING TO DATETIME')
    # Convert ride_date to datetime
    ride_sharing['ride_date'] = pd.to_datetime(ride_sharing['ride_date'])
    
    # Save today's date
    today = pd.to_datetime(dt.date.today())
    
    print('---------------------------------------SET THE RANGE')
    # Set all in the future to today's date
    print("{} to update.".format(ride_sharing.loc[ride_sharing['ride_date'] > today, 'ride_date'].shape[0]))
    ride_sharing.loc[ride_sharing['ride_date'] > today, 'ride_date'] = today
    
    # Assert change has been done
    try: assert ride_sharing['ride_date'].max().date() == today
    except: print("Error: Date greater than today!...")
    
    print('-------------------------------------------EXPLORING')
    #print(ride_sharing.info()) 
    print(ride_sharing.head(2)) 
    
    
    
def Uniqueness_constraints(seed=SEED):
    print("\n\n****************************************************")
    topic = "8. Uniqueness constraints"; print("** %s" % topic)
    
    print('-------------------------------------------EXPLORING')
    print(mlb_players.info()) 
    #print(mlb_players.head(2)) 
    
    print('----------------------------------FINDING DUPLICATES')
    # Get duplicates across all columns
    duplicates = mlb_players.duplicated()
    
    print("---------------------IDENTICAL - BASIC MASK (*)-----")
    duplicates = mlb_players.duplicated()
    print(mlb_players[duplicates]) 
    
    print("--------KEEP FALSE - IDENTICAL - BASIC MASK (*)-----")
    duplicates = mlb_players.duplicated(keep=False)
    print(mlb_players[duplicates].sort_values(by='Name')) 
    
    print("--------------------------CHECK SUBSET MASK (1)-----")
    # Column names to check for duplication
    column_names = ['Name','Team','Position']
    duplicates = mlb_players.duplicated(subset = column_names, keep = False)
    print(mlb_players[duplicates].sort_values(by='Name')) 
    
    print("-----------------DELETE IDENTICAL DUPLICATE (2)-----")
    mlb_players.drop_duplicates(inplace=True)
    duplicates = mlb_players.duplicated(subset = column_names, keep = False)
    print(mlb_players[duplicates].sort_values(by='Name')) 
    
    print("------------------------CONCILIATE THE REST (3)-----")
    # Group by column names and produce statistical summaries
    summaries = {'Height(inches)': 'max', 'Weight(lbs)': 'mean', 'Age': 'max'}
    mlb_cleaned = mlb_players.groupby(by = column_names).agg(summaries).reset_index()
    size_of_duplicated = mlb_cleaned.duplicated(subset = column_names, keep = False).sum()
    print('{} duplicated rows found!...'.format(size_of_duplicated)) 
    
    print('-------------------------------------------EXPLORING')
    print(mlb_cleaned.head(2)) 
    
    
    
def Finding_duplicates(seed=SEED):
    print("\n\n****************************************************")
    topic = "10. Finding duplicates"; print("** %s" % topic)
    
    # Prepare the data
    ride_sharing = ride_duplicated
    
    print('-------------------------------------------EXPLORING')
    print(ride_sharing.info()) 
    #print(mlb_players.head(2)) 
    
    print('----------------------------------FINDING DUPLICATES')
    # Find duplicates
    duplicates = ride_sharing.duplicated(subset=['ride_id'], keep=False)
    
    # Sort your duplicated rides
    duplicated_rides = ride_sharing[duplicates].sort_values(by='ride_id')
    
    # Print relevant columns of duplicated_rides
    print(duplicated_rides[['ride_id','duration','user_birth_year']])
    
    print('-------------------------------------------EXPLORING')
    print(duplicated_rides)
    
    
    
def Treating_duplicates(seed=SEED):
    print("\n\n****************************************************")
    topic = "11. Treating duplicates"; print("** %s" % topic)
    
    # Prepare the data
    ride_sharing = ride_duplicated
    
    print('----------------------------------FINDING DUPLICATES')
    # Find duplicates
    duplicates = ride_sharing.duplicated(subset=['ride_id'], keep=False)
    duplicates_ride_id = ride_sharing[duplicates].ride_id
    
    print('----------------------------DROPPING DUPLICATED ROWS')
    # Drop commplete duplicates from ride_sharing
    ride_dup = ride_sharing.drop_duplicates()
    
    print('--------------MAKING AGGREATION WITH DUPLICATED ROWS')
    # Create statistics dictionary for aggregation function
    statistics = {'user_birth_year': 'max', 'duration': 'mean'}
    
    # Group by ride_id and compute new statistics
    ride_unique = ride_dup.groupby('ride_id').agg(statistics).reset_index()
    
    # Find duplicated values again
    duplicates = ride_unique.duplicated(subset = 'ride_id', keep = False)
    duplicated_rides = ride_unique[duplicates == True]
    
    # Assert duplicates are processed
    try: 
        assert duplicated_rides.shape[0] == 0
        print('All duplicated rows treated, no more duplicated!...')
    except: print('Error: More duplicates!...')
    
    print('-------------------------------------------EXPLORING')
    print('reviewing treated data: \n{}'.format(
        ride_unique[ride_unique.ride_id.isin(duplicates_ride_id)][['ride_id','duration','user_birth_year']]))
    
    

def main(seed=SEED):
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Data_type_constraints()
    Numeric_data_or()
    Summing_strings_and_concatenating_numbers()
    Data_range_constraints()
    Tire_size_constraints()
    Back_to_the_future()
    Uniqueness_constraints()
    Finding_duplicates()
    Treating_duplicates()
        
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()