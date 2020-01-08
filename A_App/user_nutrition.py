# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:32:31 2019

@author: sonik
"""


import sys
import time

import requests
import pandas as pd
import os
import numpy as np


name_dicc = {
    '208': 'Energy (Kcal)',
    '203': 'Protein(g)',
    '204': 'Total Lipid (g)',
    '255': 'Water (g)',
    '307': 'Sodium(mg)',
    '269': 'Total Sugar(g)',
    '291': 'Fiber(g)',
    '301': 'Calcium(mg)',
    '303': 'Iron (mg)',
    }

class nutri:
    
     def __init__(user_df):
        
         
         prod_data = giveElements()
         print(prod_data.info())
         
       

def giveElements():
    
    
    path = "C:/Users/sonik/Documents/Advance Research/instacart_2017/step-wise-analysis/New folder/link.csv"
    link_data = pd.read_csv( path, sep=",")
    
    
    return link_data


#numbers = link['ndbno'].tolist()
#
#print(numbers)
#
#def chunks(l, n):
#    
#    for i in range(0, len(l), n):
#       
#        yield l[i:i + n]
#
#
#response = {
#        'api_key': api_key,
#        'ndbno': ['45310337','45139481'],
#        'format': 'json',
#    }
#
#req = requests.get('https://api.nal.usda.gov/ndb/V2/reports', response)
#print(req.json())
#
#
#
#arr= []
#g100 = []
#for fd in req.json()['foods']:
#        if 'food' not in fd:
#            continue
#        food = fd['food']
#        name = food['desc']['name']
#        ndbno = food['desc']['ndbno']
#        nut_dicc = {
#            '208': np.nan,
#            '203': np.nan,
#            '204': np.nan,
#            '255': np.nan,
#            '307': np.nan,
#            '269': np.nan,
#            '291': np.nan,
#            '301': np.nan,
#            '303': np.nan,
#        }
#        
#            
#
#        for nutrient in food['nutrients']:
#           
#            if nutrient['nutrient_id'] in nut_dicc:
#                
#                
#                try:
#                        print(nutrient['name'])
#                        print(nutrient['measures'][0])
#                       
#                        measure = nutrient['measures'][0]
#                        print(name)
#                       # nut_dicc[nutrient['nutrient_id']] = float(measure['value'])
#                       # nut_dicc[nutrient['nutrient_id']]
#                except:
#                    print(ndbno)
#                    sys.exit(1)
#            
#            
#               
#             
#
#        ans = {'NDB_No': ndbno, 'USDA Name': name}
#        for key, value in nut_dicc.items():
#            ans[name_dicc[key]] = value
#        arr += [ans]
#        time.sleep(1)
#
#df = pd.DataFrame(arr)
#df = df[['NDB_No', 'USDA Name', 'Energy (Kcal)', 'Total Sugar(g)', 'Total Lipid (g)', 'Water (g)', 'Protein(g)',
#         'Sodium(mg)', 'Fiber(g)', 'Calcium(mg)', 'Iron (mg)', ]]
#df.head()
#
#print(len(g100))
#print(g100)
#
##links
##https://heartbeat.fritz.ai/recommender-systems-with-python-part-ii-collaborative-filtering-k-nearest-neighbors-algorithm-c8dcd5fd89b2
#
