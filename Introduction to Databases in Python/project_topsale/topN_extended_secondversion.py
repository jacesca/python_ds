# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:57:12 2020

@author: jacesca@gmail.com
"""
###############################################################################
##  Import all dependencies.
###############################################################################
import pandas as pd
from sqlalchemy import create_engine
from glob import glob
from datetime import datetime

###############################################################################
##  Global parameters.
###############################################################################
topN = str(input()).strip() #N for top sales
table_sql = 'pbt_sales_top' #Name of the table to insert
table_comment = 'pbt_sales_top_historic' #Name of the comments' table
file = 'daily_sales.xlsx' #Name of the file to read

###############################################################################
##  User function.
###############################################################################
def create_db_connection():
    """
    To create the conection with the database
    """
    driver = 'mssql+pyodbc://'
    user_and_pwd = 'AppPBI:Vastago20'
    host_and_port = '@blinktest.database.windows.net:1433/'
    database = 'db_powerbi_test'
    engine_driver = '?driver=SQL Server'
    # Create an engine to the census database
    engine = create_engine(driver + user_and_pwd + host_and_port + database + engine_driver)
    return engine

###############################################################################
##  Create conection
###############################################################################
engine = create_db_connection()
        
###############################################################################
##  Verificating the existent file.
###############################################################################
filenames = glob(file)

if len(filenames)==1:
    ###########################################################################
    ##  Read and prepare the data.
    ###########################################################################
    sales = pd.read_excel(file, sheet_name='Sheet1', usecols = ['Date','Material','Cut GS'], date_parser = 'Date')
    max_date = sales.Date.max()
    sales = sales[sales.Date == max_date]
    sales = sales.groupby(['Date','Material'])[['Cut GS']].sum().sort_values(by=['Date','Cut GS'], ascending=False).reset_index().head(int(topN))
    sales.columns = ['pst_SalesDate', 'pst_Material', 'pst_CutGS']
    sales['pst_Rank'] = sales['pst_CutGS'].rank(ascending=False)
    
    if len(sales)>0:
        #######################################################################
        ##  Validating the date of file
        #######################################################################
        results = engine.execute("SELECT max(pst_SalesDate) FROM {}".format(table_sql)).fetchall() 
        results = results[0][0] if results[0][0] != None else '1900-01-01'
        if datetime.strptime(results, '%Y-%m-%d') < max_date:
        
            ###################################################################
            ##  Reading the comments' table
            ###################################################################
            stmt = "SELECT psh_SalesDate, psh_Material as pst_Material, psh_Comment as pst_Comment FROM {}".format(table_comment) # Build select statement for census table: stmt
            df_comment = pd.read_sql(stmt, engine)
            
            df_comment.set_index('psh_SalesDate', inplace=True)
            df_comment = df_comment.groupby('pst_Material')['pst_Comment'].last().reset_index()
            
            sales = pd.merge(sales, df_comment, how='left', on='pst_Material')
            
            ###############################################################################
            ##  Write to a mysql database.
            ###############################################################################
            # Delete data from the table_sql
            engine.execute("delete from {}".format(table_sql))
            
            # Insert data in the table_sql
            sales.to_sql(table_sql, con=engine, if_exists='append', index=False)
            
            ###########################################################################
            ##  Write to excel file.
            ###########################################################################
            sales.head(int(topN)).to_excel('daily_top_sales.xlsx')

###################################################################
##  Retrieve the information from top rank table
###################################################################
stmt = "SELECT * FROM {}".format(table_sql) # Build select statement for census table: stmt
df_sales = pd.read_sql(stmt, engine)
print(df_sales)            