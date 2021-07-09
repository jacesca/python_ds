# -*- coding: utf-8 -*-
"""
Created on Tue May  7 05:40:28 2019

@author: jacqueline.cortez
Este programa lee un archivo pdf, limpia los datos y los salva en formato csv.

Encoding
https://docs.python.org/3/library/codecs.html#standard-encodings

Documentation:
    https://pypi.org/project/tabula-py/
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Panel.to_csv.html?highlight=to_csv#pandas.Panel.to_csv

"""

import tabula 
import pandas as pd

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
print("READING THE FILE")
#df = read_pdf('C:\\Users\\sreeraj.va\\Desktop\\kkrvspnj.pdf')--> Using a long path
file = "data_fifa_player_list.pdf"
df = tabula.read_pdf(file, pages="all")

print("Exploring the information found...")
print(df.shape)
print(df.head())
print(df.tail())
print(df.columns)


print("\n****************************************************")
print("CLEANING THE DATA")

print("Deleting multiple header from many pages...")
# Delete duplicate rows from the header
rows_to_delete = df[df["Pos. FIFA Popular Name"]=="Pos. FIFA Popular Name"].index
df.drop(labels=rows_to_delete, axis="rows", inplace=True) #axis=0


print("\nSplitting 'Pos. FIFA Popular Name' column into 'Position' and 'FIFA name'...")
# Split the column 'Birth Date Shirt Name'
new_column = df["Pos. FIFA Popular Name"].str.split(" ", n=1, expand=True)
df["Position"] = new_column[0] 
df['FIFA name'] = new_column[1] 


print("\nSplitting 'Birth Date Shirt Name' column into 'Birth Date' and 'Name'...")
# Split the column 'Birth Date Shirt Name'
new_column = df["Birth Date Shirt Name"].str.split(" ", n=1, expand=True)
df["birth date"] = pd.to_datetime(new_column[0]) #Convert to datetime columns
df['name'] = new_column[1] #Getting the name column


print("\nSplitting 'Height Weight' column into 'Height' and 'Weight'...")
# Split the column 'Height Weight'
new_column = df["Height Weight"].str.split(" ", n=1, expand=True)
df["Height"] = new_column[0] 
df['Weight'] = new_column[1] 


print("\nFixing index...")
df.reset_index(inplace=True)


print("\nDropping columns 'Pos. FIFA Popular Name', 'Unnamed: 4', 'Birth Date Shirt Name' and 'Height Weight'...")
#Drops unnecesary columns
df.drop(['index', 'Pos. FIFA Popular Name', 'Unnamed: 4', 'Birth Date Shirt Name', 'Height Weight'], axis='columns', inplace=True)


print("\nFixing names columns...")
df.columns = ['team', 'player_number', 'club', 'position', 'fifa_name', 'birth', 'name', 'height', 'weight']


print("\nFinal result")
print(df.shape)
print(df.head())

print("\n****************************************************")
print("SAVED INTO CSV FILE\n")
file_out = "data_fifa_player_list.csv"
#df.to_csv(file_out, encoding = "utf_32")
df.to_csv(file_out)

print("****************************************************")
print("** END                                            **")
print("****************************************************")

"""
How to extract more than one table present in a pdf file with tabula in python?
https://stackoverflow.com/questions/49733576/how-to-extract-more-than-one-table-present-in-a-pdf-file-with-tabula-in-python
Code example:
    
import os
from tabula import wrapper
os.chdir("E:/Documents/myPy/")
tables = wrapper.read_pdf("MyPDF.pdf",multiple_tables=True,pages='all',encoding='utf-8',spreadsheet=True)

i=1
for table in tables:
    table.columns = table.iloc[0]
    table = table.reindex(table.index.drop(0)).reset_index(drop=True)
    table.columns.name = None
    #To write Excel
    table.to_excel('output'+str(i)+'.xlsx',header=True,index=False)
    #To write CSV
    table.to_csv('output'+str(i)+'.csv',sep='|',header=True,index=False)
    i=i+1
"""

###########################################################
## DOCUMENTATION
## 
## tabula-py package:
## https://pypi.org/project/tabula-py/
## https://tabula.technology/
## https://www.ikkaro.com/convertir-pdf-excel-csv/
##
## camelot package:
## https://camelot-py.readthedocs.io/en/master/
##
## COMPARISION BETWEEN PACKAGE FOR READING PDF 
## https://github.com/socialcopsdev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools
##
##
## ERRORS :
## ModuleNotFoundError: No module named 'tabula'
## You have to uso de Anaconda Prompt
## https://stackoverflow.com/questions/52175067/tabula-pip-installer-says-successful-download-but-unable-to-import
##
## ModuleNotFoundError: No module named 'cv2' "tabula'
## https://stackoverflow.com/questions/19876079/cannot-find-module-cv2-when-using-opencv
## https://stackoverflow.com/questions/19876079/cannot-find-module-cv2-when-using-opencv/41895783#41895783
##
## JavaNotFoundError(JAVA_NOT_FOUND_ERROR) "tabula"
## https://www.java.com/es/download/
## https://www.quora.com/What-is-the-solution-for-getting-the-Java-not-found-error-in-Talend-or-Netbeans-or-related-applications-running-over-JVM
## https://javatutorial.net/set-java-home-windows-10
## https://confluence.atlassian.com/doc/setting-the-java_home-variable-in-windows-8895.html
"""
Set the JAVA_HOME Variable
To set the JRE_HOME or JAVA_HOME variable:
    1. Locate your Java installation directory
        If you didn't change the path during installation, 
        it'll be something like C:\Program Files\Java\jdk1.8.0_65
        
        You can also type where java at the command prompt.
    2. Do one of the following:
        Windows 7 – Right click My Computer and select Properties > Advanced 
        Windows 8 – Go to Control Panel > System > Advanced System Settings
        Windows 10 – Search for Environment Variables then select Edit the system environment variables
    3. Click the Environment Variables button.
    4. Under System Variables, click New.
    5. In the Variable Name field, enter either:
        JAVA_HOME if you installed the JDK (Java Development Kit)
        or
        JRE_HOME if you installed the JRE (Java Runtime Environment) 
    6. In the Variable Value field, enter your JDK or JRE installation path .
    7. Click OK and Apply Changes as prompted
"""
##
## RuntimeError: Please make sure that Ghostscript is installed "camelot"
## https://www.ghostscript.com/download/gsdnld.html
## 
## OSError: exception: access violation writing 0x0D8AB8C0 "camelot"
##
## INSTALED LIBRARIES:
## pip install tabula-py
## https://www.java.com/es/download/
##
## pip install camelot
## pip install opencv-python
## 
## conda update anaconda-navigator
## conda update navigator-updater
