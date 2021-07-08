# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:45:14 2019

@author: jacqueline.cortez
"""

import random
import numpy as np

print("Ingrese cuantos numeros aleatorios desea obtener")
n=int(input())

aleatorios1 = [random.randint(0,1000) for _ in range(n)]
aleatorios2 = np.random.rand(n) #height
aleatorios3 = np.random.choice(np.arange(1000), n) #age
aleatorios4 = np.random.randint(50,100, size=n) #age
aleatorios5 = np.random.randint(1000, size=n) #age

print(aleatorios1)
print(aleatorios2)
print(aleatorios3)
print(aleatorios4)
print(aleatorios5)