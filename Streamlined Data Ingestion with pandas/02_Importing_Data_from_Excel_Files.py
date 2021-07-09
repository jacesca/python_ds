# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 2: Importing Data From Excel Files
    Practice using pandas to get just the data you want from flat files, learn 
    how to wrangle data types and handle errors, and look into some U.S. tax data 
    along the way.
Source: https://learn.datacamp.com/courses/streamlined-data-ingestion-with-pandas
Help: https://strftime.org/
"""
###############################################################################
## Importing libraries
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt



###############################################################################
## Preparing the environment
###############################################################################
# Global variables
excel_file = 'fcc-new-coder-survey.xlsx'

# Global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 6, 'font.size': 8})
suptitle_param = dict(color='darkblue', fontsize=9)
title_param    = {'color': 'darkred', 'fontsize': 10}



###############################################################################
## Main part of the code
###############################################################################
def Introduction_to_spreadsheets():
    print("****************************************************")
    topic = "1. Introduction to spreadsheets"; print("** %s" % topic)
    print("****************************************************")
    topic = "2. Get data from a spreadsheet"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Loading Spreadsheets')
    # Read the Excel file
    survey_data = pd.read_excel(excel_file)
    # View the first 5 lines of data
    print('Shape: ', survey_data.shape)
    print(survey_data.head())
    
    print('---------------------------------------------Loading Select Columns and Rows')
    # Read columns W-AB and AR of file, skipping metadata header
    survey_data = pd.read_excel(excel_file,
                                skiprows=2, usecols="W:AB, AR")
    # View data
    print(survey_data.head())
    
    
    
def Load_a_portion_of_a_spreadsheet():
    print("****************************************************")
    topic = "3. Load a portion of a spreadsheet"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Loading Select Columns and Rows')
    # Create string of lettered columns to load
    col_string = 'AD, AW:BA'
    # Load data with skiprows and usecols set
    survey_responses = pd.read_excel(excel_file, skiprows=2, usecols=col_string)
    # View the names of the columns selected
    print(survey_responses.columns)
    
    
    
def Getting_data_from_multiple_worksheets():
    print("****************************************************")
    topic = "4. Getting data from multiple worksheets"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Loading Select Sheets')
    # Create string of lettered columns to load
    col_string = 'AD, AW:BA'
    # Get the second sheet by position index
    survey_data_sheet2 = pd.read_excel(excel_file, sheet_name=1, 
                                       skiprows=2, usecols=col_string)
    # Get the second sheet by name
    survey_data_2017 = pd.read_excel(excel_file, sheet_name='2017')
    print("data load with number sheet and name sheet are Equal?: ", survey_data_sheet2.equals(survey_data_2017))
    print(survey_data_2017.head())
    
    print('---------------------------------------------Loading All Sheets')
    survey_data = pd.read_excel(excel_file, sheet_name=None, 
                                skiprows=2, usecols=col_string)
    #survey_data = pd.read_excel(excel_file, sheet_name=None)
    print(type(survey_data))
    for year, df in survey_data.items():
        print(year, type(df), 'Shape: ', df.shape)
    
    print('---------------------------------------------Putting It All Together')
    # Create empty data frame to hold all loaded sheets
    all_responses = pd.DataFrame()
    # Iterate through data frames in dictionary
    for sheet_name, frame in survey_data.items():
        #Preview the data
        print(f'Head of {sheet_name}: \n{frame.head()}')
        # Add a column so we know which year data is from
        frame["Year"] = sheet_name
        # Add each data frame to all_responses
        all_responses = all_responses.append(frame)
    # View years in data
    print(all_responses.Year.unique(), "shape: ", all_responses.shape)
    #Preview the data
    print(f'Head of all_responses: \n{all_responses.head()}')
        
    
    
def Select_a_single_sheet():
    print("****************************************************")
    topic = "5. Select a single sheet"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Loading Spreadsheets')
    # Create df from second worksheet by referencing its position
    responses_2017_colname = pd.read_excel(excel_file, sheet_name=1, 
                                           skiprows=2, usecols=['JobPref'])
    # Create df from second worksheet by referencing its name
    responses_2017_numname = pd.read_excel(excel_file, sheet_name='2017', 
                                           skiprows=2, usecols=['JobPref'])
    print("data load with number sheet and name sheet are Equal?: ", 
          responses_2017_colname.equals(responses_2017_numname))
    
    print('---------------------------------------------Plot the job preference')
    # Graph where people would like to get a developer job
    job_prefs = responses_2017_colname.groupby("JobPref").JobPref.count()
    
    fig, ax = plt.subplots()
    job_prefs.plot.barh()
    ax.set_xlabel('Frequency')
    ax.set_title('Job Preferences', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.4, bottom=None, right=.9, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Select_multiple_sheets():
    print("****************************************************")
    topic = "6. Select multiple sheets"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Loadd by sheet name list')
    # Load both the 2016 and 2017 sheets by name
    all_survey_data = pd.read_excel(excel_file, sheet_name=['2016', '2017'])
    # View the data type of all_survey_data
    print(all_survey_data.keys())
    
    print('---------------------------------------------Loadd by sheet number list')
    # Load both the 2016 and 2017 sheets by name
    all_survey_data = pd.read_excel(excel_file, sheet_name=[0, 1])
    # View the data type of all_survey_data
    print(all_survey_data.keys())
    
    print('---------------------------------------------Load by mix list')
    # Load all sheets in the Excel file
    all_survey_data = pd.read_excel(excel_file, sheet_name = [0, '2017'])
    # View the sheet names in all_survey_data
    print(all_survey_data.keys())
    
    print('---------------------------------------------Load everything with None')
    # Load all sheets in the Excel file
    all_survey_data = pd.read_excel(excel_file, sheet_name=None)
    # View the sheet names in all_survey_data
    print(all_survey_data.keys())
    
    
    
def Work_with_multiple_spreadsheets():
    print("****************************************************")
    topic = "7. Work with multiple spreadsheets"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Join both sheet names')
    # Loading All Sheets
    responses = pd.read_excel(excel_file, sheet_name=None, 
                              skiprows=2, usecols=['EmploymentStatus'])
    # Create an empty data frame
    all_responses = pd.DataFrame()
    
    # Set up for loop to iterate through values in responses
    for df in responses.values():
        # Print the number of rows being added
        print("Adding {} rows".format(df.shape[0]))
        # Append df to all_responses, assign result
        all_responses = all_responses.append(df)
        
    print('---------------------------------------------Plot the employment status')
    # Graph employment statuses in sample
    counts = all_responses.groupby("EmploymentStatus").EmploymentStatus.count()
    
    fig, ax = plt.subplots()
    counts.plot.barh()
    ax.set_xlabel('Frequency')
    ax.set_title('Employment Status', **title_param)    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.4, bottom=None, right=.9, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Modifying_imports_true_false_data():
    print("****************************************************")
    topic = "8. Modifying imports: true/false data"; print("** %s" % topic)
    print("****************************************************")
    print('---------------------------------------------Preparing the configuration')
    pd.set_option("display.max_columns",20)

    print('---------------------------------------------Pandas and Booleans (default load)')
    columns_name = ['AttendedBootcamp', 'BootcampFinish', 'BootcampLoanYesNo', 
                    'HasDebt', 'HasFinancialDependents', 'HasHighSpdInternet']
    bootcamp_data = pd.read_excel(excel_file, sheet_name='2016', skiprows=2, 
                                  nrows=100, usecols=columns_name)
    print('shape: ', bootcamp_data.shape)
    print(bootcamp_data.dtypes)
    print(bootcamp_data.head())
    for col in columns_name:
        print(f'Unique values of {col} :', bootcamp_data[col].unique())
    
    print('---------------------------------------------Counting null values')
    # Count NAs
    print(bootcamp_data.isna().sum())
    
    print('---------------------------------------------Booleans (Try boolean/bool type to load)')
    # Create dict specifying data types
    data_types   = {"AttendedBootcamp"      : "boolean", # This type accept Null values, but it is only work on True/False literal strings, not 0/1 or Yes/No.
                    'HasDebt'               : bool, #Only works with columns without null values
                    'HasFinancialDependents': bool} #When column has strings, convert everything to true, it doesn't work well
    bool_data = pd.read_excel(excel_file, sheet_name='2016', skiprows=2, 
                              nrows=100, usecols=columns_name,
                              dtype=data_types)
    print(bool_data.dtypes)
    print('\nThere is a mismatch in "HasFinancialDependents" column. Take a look:')
    print(bool_data.head())
        
    print('---------------------------------------------Booleans (Try dtype with true_values/false_values params)')
    # Create dict specifying true_values
    bool_data = pd.read_excel(excel_file, sheet_name='2016', skiprows=2, 
                              nrows=100, usecols=columns_name,
                              dtype=data_types,
                              true_values=['Yes'], false_values=['No'])
    print(bool_data.dtypes)
    print('\nBetter! Now there is no mismatch in "HasFinancialDependents" column:')
    print(bool_data.head())
        
    print('---------------------------------------------Setting the default configuration')
    pd.reset_option("display.max_columns")
    
    
    
def Set_Boolean_columns():
    print("****************************************************")
    topic = "9. Set Boolean columns"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Preparing the configuration')
    pd.set_option("display.max_columns",20)

    print('---------------------------------------------Explore')
    # Set the columns to read
    columns_name = ['ID.x', 'HasDebt', 'HasFinancialDependents', 
                    'HasHomeMortgage', 'HasStudentDebt']
    # Load the data
    survey_data = pd.read_excel(excel_file, sheet_name='2016', skiprows=2, 
                                nrows=100, usecols=columns_name)
    
    # Count NA values in each column
    print(survey_data.info())
    print(survey_data.head())
    
    print('---------------------------------------------Setting bool columns')
    print("Setting only the columns that does not have null values.")
    # Set dtype to load appropriate column(s) as Boolean data
    survey_data = pd.read_excel(excel_file, sheet_name='2016', skiprows=2, 
                                nrows=100, usecols=columns_name, 
                                dtype = {'HasDebt': bool,
                                         'HasFinancialDependents': bool},
                                true_values=['Yes'], false_values=['No'])
    print(survey_data.info())
    
    print('---------------------------------------------View financial burdens')
    # View financial burdens by Boolean group
    print(survey_data.groupby('HasDebt').sum())
    
    print('---------------------------------------------Setting the default configuration')
    pd.reset_option("display.max_columns")
    
    
    
def Set_custom_true_false_values():
    print("****************************************************")
    topic = "10. Set custom true/false values"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Preparing the configuration')
    pd.set_option("display.max_columns",20)

    print('---------------------------------------------Explore')
    # Set the columns to read
    columns_name = ['ID.x', 'HasDebt', 'HasFinancialDependents', 
                    'HasHomeMortgage', 'HasStudentDebt']
    # Explore the file first
    survey_subset = pd.read_excel(excel_file, sheet_name='2016', 
                                  skiprows=2, nrows=100, usecols=columns_name)
    print(survey_subset.head())
    
    print('---------------------------------------------Set Bool columns')
    # Load file with Yes as a True value and No as a False value
    survey_subset = pd.read_excel(excel_file, sheet_name='2016', 
                                  skiprows=2, nrows=100, usecols=columns_name, 
                                  dtype={"HasDebt": bool, "AttendedBootCampYesNo": bool},
                                  true_values=['Yes'], false_values=['No'])
    # View the data
    print(survey_subset.head())
    
    print('---------------------------------------------Setting the default configuration')
    pd.reset_option("display.max_columns")
    
    
    
def Modifying_imports_parsing_dates():
    print("****************************************************")
    topic = "11. Modifying imports: parsing dates"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Preparing the configuration')
    pd.set_option("display.max_columns",20)

    print('---------------------------------------------Explore')
    # Set the columns to read
    columns_name = ['Part1EndTime', 'Part1StartTime', 
                    'Part2EndTime', 'Part2StartDate', 'Part2StartTime']
    # Load file, parsing standard datetime columns
    survey_df = pd.read_excel(excel_file, skiprows=2, usecols=columns_name)
    # Check data types of timestamp columns
    print(survey_df.dtypes)
    print(survey_df.head())
    
    print('---------------------------------------------Parsing Dates')
    # Set the columns to read
    columns_name = ['Part1EndTime', 'Part1StartTime', 
                    'Part2EndTime', 'Part2StartDate', 'Part2StartTime']
    # List columns of dates to parse
    date_cols = ['Part1EndTime', 'Part1StartTime']
    # Load file, parsing standard datetime columns
    survey_df = pd.read_excel(excel_file, skiprows=2, usecols=columns_name, 
                              parse_dates=date_cols)
    # Check data types of timestamp columns
    print(survey_df.dtypes)
    print(survey_df.head())
    
    print('---------------------------------------------Combining columns')
    # List columns of dates to parse
    date_cols = ['Part1EndTime', 'Part1StartTime',
                 ['Part2StartDate', 'Part2StartTime']]
    # Load file, parsing standard and split datetime columns
    survey_df = pd.read_excel(excel_file, skiprows=2, usecols=columns_name, 
                              parse_dates=date_cols)
    # Check data types of timestamp columns
    print(survey_df.dtypes)
    print(survey_df.head())
    
    print('---------------------------------------------Renaming the combine columns')
    # List columns of dates to parse
    date_cols = {'Part1End'    : ['Part1EndTime'], 
                 'Part1Start'  : ['Part1StartTime'],
                 'Part2Start'  : ['Part2StartDate', 'Part2StartTime']}
    # Load file, parsing standard and split datetime columns
    survey_df = pd.read_excel(excel_file, skiprows=2, usecols=columns_name, 
                              parse_dates=date_cols)
    # Check data types of timestamp columns
    print(survey_df.dtypes)
    print(survey_df.head())
    
    print('---------------------------------------------Parsing Non-Standard Dates')
    format_string = "%Y%m%d %H:%M:%S"
    survey_df["Part2EndTime"] = pd.to_datetime(survey_df["Part2EndTime"],
                                               format=format_string)
    # Check data types of timestamp columns
    print(survey_df.dtypes)
    print(survey_df.head())
    
    print('---------------------------------------------Setting the default configuration')
    pd.reset_option("display.max_columns")
    
    
    
def Parse_simple_dates():
    print("****************************************************")
    topic = "12. Parse simple dates"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Parsing datetime columns')
    # Set the columns to read
    columns_name = ['Part1StartTime']
    # Load file, with Part1StartTime parsed as datetime data
    survey_data = pd.read_excel(excel_file, skiprows=2, usecols=columns_name, 
                                parse_dates=['Part1StartTime'])

    # Print first few values of Part1StartTime
    print(survey_data.Part1StartTime.head())

    
    
def Get_datetimes_from_multiple_columns():
    print("****************************************************")
    topic = "13. Get datetimes from multiple columns"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Parsing datetime dict')
    # Set the columns to read
    columns_name = ['Part2StartDate', 'Part2StartTime']
    # Create dict of columns to combine into new datetime column
    datetime_cols = {"Part2Start": ['Part2StartDate', 'Part2StartTime']}
    # Load file, supplying the dict to parse_dates
    survey_data = pd.read_excel(excel_file, skiprows=2, usecols=columns_name, 
                                parse_dates=datetime_cols)
    # View summary statistics about Part2Start
    print(survey_data.Part2Start.describe())
        
    
def Parse_non_standard_date_formats():
    print("****************************************************")
    topic = "14. Parse non-standard date formats"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Explore')
    print('---------------------------------------------Explore')
    
    
        
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_spreadsheets()
    Load_a_portion_of_a_spreadsheet()
    Getting_data_from_multiple_worksheets()
    Select_a_single_sheet()
    Select_multiple_sheets()
    Work_with_multiple_spreadsheets()
    Modifying_imports_true_false_data()
    Set_Boolean_columns()
    Set_custom_true_false_values()
    Modifying_imports_parsing_dates()
    Parse_simple_dates()
    Get_datetimes_from_multiple_columns()
    Parse_non_standard_date_formats()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    plt.style.use('default')