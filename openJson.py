# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:21:49 2016

@author: Shuxian
"""

import json

data = []
with open('decistion tree_6.json') as f:
    for line in f:
        data.append(json.loads(line))
        
data1=json.load(open('decistion tree_6.json'))
print(data1)
        

import pandas as pd
df = pd.read_csv('decistion tree_6.csv') 
df
        