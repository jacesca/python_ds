# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:57:12 2020

@author: jacesca@gmail.com
"""
###############################################################################
##  Import all dependencies.
###############################################################################
import pandas as pd
import pprint

from sqlalchemy import create_engine
from sqlalchemy import Table
from sqlalchemy import MetaData
from glob import glob

###############################################################################
##  Verificating the existent file.
###############################################################################
file = 'daily_sales.xlsx'
filenames = glob(file)

if len(filenames)==1:
    ###############################################################################
    ##  Reading external variables.
    ###############################################################################
    topN = str(input()).strip()


    ###############################################################################
    ##  Read and prepare the data.
    ###############################################################################
    sales = pd.read_excel(file, sheet_name='Sheet1', usecols = ['Date','Material','Cut GS'], date_parser = 'Date')
    sales = sales[sales.Date == sales.Date.max()]
    sales = sales.groupby(['Date','Material'])[['Cut GS']].sum().sort_values(by=['Date','Cut GS'], ascending=False).reset_index().head(int(topN))
    sales.columns = ['pst_SalesDate', 'pst_Material', 'pst_CutGS']
    sales['pst_Rank'] = sales['pst_CutGS'].rank(ascending=False)
    print(sales)

    ###############################################################################
    ##  Write to excel file.
    ###############################################################################
    sales.head(int(topN)).to_excel('daily_top_sales.xlsx')
    
    """
    ###############################################################################
    ##  Prepare the connection to mysql database.
    ###############################################################################
    driver = 'mysql+pymysql://'
    user_and_pwd = 'AppPBI:Vastago20'
    host_and_port = '@35.230.92.36:3306/'
    database = 'db_powerbi_test'
    table_sql = 'db_pbt_sales_top'
    
    # Create an engine to the census database
    engine = create_engine(driver + user_and_pwd + host_and_port + database)
    """
    
    ###############################################################################
    ##  Prepare the connection to Microsoft SQL database. --> https://medium.com/@anushkamehra16/connecting-to-sql-database-using-sqlalchemy-in-python-2be2cf883f85
    ###############################################################################
    driver = 'mssql+pyodbc://'
    user_and_pwd = 'AppPBI:Vastago20'
    host_and_port = '@blinktest.database.windows.net:1433/'
    database = 'db_powerbi_test'
    engine_driver = '?driver=SQL Server'
    table_sql = 'pbt_sales_top'
    
    # Create an engine to the census database
    engine = create_engine(driver + user_and_pwd + host_and_port + database + engine_driver)
    
    
    ###############################################################################
    ##  Knowing the table we will work with
    ###############################################################################
    # Prepare the conection to work with internal object in database
    metadata = MetaData() 
    
    # Reflect the table_sql from the engine
    table_retrieve = Table(table_sql, metadata, autoload=True, autoload_with=engine) 
    
    # Print the column names
    print("Columns name: \n{}\n".format(table_retrieve.columns.keys())) 
    
    # Print full table metadata
    print("Metadata: \n{}\n\n".format(repr(metadata.tables[table_sql]))) 
    
    # First way: Execute the statement and store all the records: results
    results = engine.execute("SELECT * FROM {}".format(table_sql)).fetchall() 
    
    # Second way: 
    #connection = engine.connect() # Open the connection
    #stmt = "SELECT * FROM {}".format(table_sql) # Build select statement for census table: stmt
    #results = connection.execute(stmt).fetchall()
    #connection.close() # Close the connection
        
    if len(results)>0:
        # Print the keys/column names of the results returned
        print(results[0].keys()) 
        
        # Print the data in results returned
        pprint.pprint(results)
        
        # Create a DataFrame from the results: df
        df = pd.DataFrame(results) 
        
        # Set column names
        df.columns = results[0].keys()
        
        # Print the Dataframe
        print("Data in a dataframe: \n{}\n\n".format(df)) 
    else:
        print("No data in the table: {}".format(table_sql))

    ###############################################################################
    ##  Setting the columns name
    ###############################################################################
    #metadata = MetaData()  # Prepare the conection to work with internal object in database
    #table_retrieve = Table(table_sql, metadata, autoload=True, autoload_with=engine) # Reflect the table_sql from the engine
    #sales.columns = table_retrieve.columns.keys() # Replacing the column names
        
        
        ###############################################################################
        ##  Write to a mysql database.
        ###############################################################################
        # Delete data from the table_sql
        engine.execute("delete from {}".format(table_sql))
        
        # Insert data in the table_sql
        sales.to_sql(table_sql, con=engine, if_exists='append', index=False)
        
        
        ###############################################################################
        ##  Just reading the data from SQL after insert
        ###############################################################################
        stmt = "SELECT * FROM {}".format(table_sql) # Build select statement for census table: stmt
        df2 = pd.read_sql(stmt, engine)
        
        # Print the Dataframe
        print("Data in a dataframe after insert: \n{}\n\n".format(df2)) 
else:
    print("File does not exist.")