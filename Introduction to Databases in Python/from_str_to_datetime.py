# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:58:17 2020

@author: jacesca@gmail.com

From str to datetime
"""

from datetime import datetime

my_date = datetime.strptime("2014-06-03", '%Y-%m-%d')
print(type(my_date))
print(my_date)

my_str_date = my_date.strftime("%Y-%m-%d")
print(type(my_str_date))
print(my_str_date)