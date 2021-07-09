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


###############################################################################
##  Reading external variables.
###############################################################################
topN = str(input()).strip()


###############################################################################
##  Read and prepare the data.
###############################################################################
sales = pd.read_excel('daily_sales.xlsx', sheet_name='Sheet1', usecols = ['Date','Material','Cut GS'], date_parser = 'Date')
sales = sales.groupby(['Date','Material'])[['Cut GS']].sum().sort_values(by=['Cut GS'], ascending=False).reset_index().head(int(topN))
sales.columns = ['pst_SalesDate', 'pst_Material', 'pst_CutGS']
sales['pst_Rank'] = sales['pst_CutGS'].rank(ascending=False)
#print(sales)


###############################################################################
##  Write to excel file.
###############################################################################
#sales.head(int(topN)).to_excel('daily_top_sales.xlsx')


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


###############################################################################
##  Write to a mysql database.
###############################################################################
engine.execute("delete from {}".format(table_sql)) # Delete data from the table_sql
sales.to_sql(table_sql, con=engine, if_exists='append', index=False) # Insert data in the table_sql
