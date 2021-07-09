# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:14:08 2019

@author: jacqueline.cortez
"""
import pandas as pd
from glob import glob

pd.options.display.float_format = '{:,.0f}'.format 

#Establece los archivos a leer
filenames = glob("01.08 *.csv")
#comprobando los 51 archivos csv
#print("Numero de archivos recuperados: ",len(filenames))

#Lee los archivos seleccionados #index_col="Nombre"
dfs = [pd.read_csv(f, header=None, usecols=[1,2,3,4], names=["Genero","Desde","Nombre","Total"]) for f in filenames]

#Concatena en un solo dataframes
df = pd.concat(dfs, ignore_index=True)
#Imprime el tamaño del dataframe concatenado
#print("\nTamaño del dataframe concatenado: ",df.shape)
#Imprime las primeras 5 filas del dataframe concatenado
#print("\nPrimeras 2 filas del dataframe concatenado:")
#print(df.head(2))

#Agrupa por Nombre y Género
df=df.groupby(["Nombre","Genero"]).agg({"Desde":min,"Total":sum})
#Imprime el tamaño del dataframe agrupado
#print("\nTamaño del dataframe agrupado: ",df.shape)
#print("\nPrimeras 2 filas del dataframe agrupado:")
#print(df.head(2))

dfm = df.loc[(slice(None),"M"),:]
#print("\nTamaño del dataframe con nombres masculinos: ", dfm.shape)
#print("\nPrimeras 2 filas del dataframe:")
#print(dfm.head(2))

dff = df.loc[(slice(None),"F"),:]
#print("\nTamaño del dataframe con nombres femeninos: ", dff.shape)
#print("\nPrimeras 2 filas del dataframe:")
#print(dff.head(2))

nameF = ["Julieta", "Fernanda", "Jacqueline", "Gabriela"]
for name in nameF:
    mask = df.index.get_level_values("Nombre")==name
    #mask = df.index.get_level_values("Nombre").str.contains(name)
    print("\n¿Cuántas veces se repitió el nombre '", name,"'? ")
    print(df.loc[(mask,"F"),:])
    #Recupera un valor específico dentro del DataFrame.
    #df.at[(mask,"F"),:]

nameM = ["Jose","Lisandro"]
for name in nameM:
    mask = df.index.get_level_values("Nombre")==name
    #mask = df.index.get_level_values("Nombre").str.contains(name)
    print("\n¿Cuántas veces se repitió el nombre '", name,"'? ")
    print(df.loc[(mask,"M"),:])

#print("\nLos 5 nombres femeninos menos comunes:")
#print(dff.sort_values("Total").head())

#print("\nLos 5 nombres femeninos más comunes:")
#print(dff.sort_values("Total",ascending=False).head())

#print("\nLos 5 nombres masculinos menos comunes:")
#print(dfm.sort_values("Total").head())

#print("\nLos 5 nombres masculinos más comunes:")
#print(dfm.sort_values("Total",ascending=False).head())