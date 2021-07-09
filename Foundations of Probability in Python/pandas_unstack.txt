# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:36:19 2020

@author: jaces
"""
import numpy as np
import pandas as pd

index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
                                   ('two', 'a'), ('two', 'b')])

s = pd.Series(np.arange(1.0, 5.0), index=index)

print("Original data: \n{}".format(s))
print("\n\n.Stack(level=-1)\n{}".format(s.unstack(level=-1)))

df = s.unstack(level=0)
print("\n\ndf = s.unstack(level=0)\n{}".format(df))
print("\n\ndf.unstack()\n{}".format(df.unstack()))