# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:32:31 2019

@author: sonik
"""


import sys
import time

import requests
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt





name_dicc = {
    '208': 'Energy (Kcal)',
    '203': 'Protein(g)',
    '204': 'Total Lipid (g)',    
    '307': 'Sodium(mg)',
    '269': 'Total Sugar(g)',
    '291': 'Fiber(g)',
    '301': 'Calcium(mg)',
    '303': 'Iron (mg)',
    }

api_key = "fTDlZCFQsnNuRFKrJoWMbW2u8bvj5vOATefxjzg1"

user_product_data = pd.DataFrame() #creates a new dataframe that's empty

nutrition_data = pd.DataFrame()
class nutrition:
    
     def __init__(self,user_df):
        
         print("### user_products##")
         prod_data = giveElements()
         
         
        
         data_o = user_df.merge(prod_data, left_on='product_id',right_on='product_id')
         user_product_data = data_o
         
         
         nutri_data = getnutrition(user_product_data)
         
        
             
         plot_chart(nutri_data)    
         
         

def plot_chart(nutri_data):
    
    foodlist = []
    for key, value in nutri_data.iteritems(): 
#        print(key) 
#        print(value)
#        print() 
        if key == "USDA Name":
            foodlist = value
    
    print()
    print()
    print()
    print()
    print("#######")
          
    print("food Ordered frequently along with Nutriotional Analysis")        
    print(foodlist)       
        
    print(nutri_data.info())
    
    print("Total Appx Energy Values" + str(nutri_data[['Energy (Kcal)']].sum()))
    
    nutri_data[['Energy (Kcal)']].sum().plot.bar()
    
  
    plt.title('Total Energy Values')
    plt.xlabel('Energy (Kcal)')
    plt.ylabel('value')
    
    
    plt.show()
    
    time.sleep(10)
    
    print("Total Appx Sugar Values" + str(nutri_data[['Total Sugar(g)']].sum()))
    nutri_data[['Total Sugar(g)']].sum().plot.bar()

    
    plt.title('Total Sugar Values')
    plt.xlabel('Total Sugar(g)')
    plt.ylabel('value')
    
    plt.show()
    
    time.sleep(10)
    
    print("Total Appx Protein(g) Values" + str(nutri_data[['Protein(g)']].sum()))
    nutri_data[['Protein(g)']].sum().plot.bar()
    
    plt.title('Total Protein Values')
    plt.xlabel('Protein(g)')
    plt.ylabel('value')
    
    
    
    plt.show()
    
    time.sleep(10)
    
    
    print("Total Appx Calcium(mg) Values" + str(nutri_data[['Calcium(mg)']].sum()))
    nutri_data[['Calcium(mg)']].sum().plot.bar()
    
    
    plt.title('Total Calcium(mg) Values')
    plt.xlabel('Calcium(mg)')
    plt.ylabel('value')
    
    
    
    plt.show()
    
    time.sleep(10)
    
    
    print("Total Appx Sodium(mg) Values" + str(nutri_data[['Sodium(mg)']].sum()))
    nutri_data[['Sodium(mg)']].sum().plot.bar()

    plt.title('Total Sodium(mg) Values')
    plt.xlabel('Sodium(mg)')
    plt.ylabel('value')
    
    plt.show()
    
    time.sleep(10)
    
    
    print("Total Appx Fiber(g) Values" + str(nutri_data[['Fiber(g)']].sum()))
    nutri_data[['Fiber(g)']].sum().plot.bar()
    
    plt.title('Total Fiber(g) Values')
    plt.xlabel('Fiber(g)')
    plt.ylabel('value')
    
    plt.show()
    
    plt.show()
    
    
        

def giveElements():
    
    
    link_data = pd.read_csv("C:/Users/sonik/Documents/Advance Research/instacart_2017/step-wise-analysis/link.csv")
     
    return link_data
    
    

def getnutrition(user_product_data):
    numbers = user_product_data['ndbno'].tolist()
    response = {
        'api_key': api_key,
        'ndbno': numbers,
        'format': 'json',
    }
    req = requests.get('https://api.nal.usda.gov/ndb/V2/reports', response)
    #print(req.json())
    arr = []
    g100 = []
    
    req = requests.get('https://api.nal.usda.gov/ndb/V2/reports', response)
   
    for fd in req.json()['foods']:
            if 'food' not in fd:
                continue
            food = fd['food']
            name = food['desc']['name']
            ndbno = food['desc']['ndbno']
            nut_dicc = {
                '208': np.nan,
                '203': np.nan,
                '204': np.nan,                
                '307': np.nan,
                '269': np.nan,
                '291': np.nan,
                '301': np.nan,
                '303': np.nan,
            }
            ver = True
    
    
    
            
            for nutrient in food['nutrients']:
                if nutrient['nutrient_id'] in nut_dicc and ('measures' not in nutrient or len(nutrient['measures']) == 0 or nutrient['measures'] == [None]):
                        ver = False
                if not ver:
                    g100 += [ndbno]
                    print(ndbno)
        
                for nutrient in food['nutrients']:
                    if nutrient['nutrient_id'] in nut_dicc:
                        try:
                            if ver:
                                measure = nutrient['measures'][0]
                                nut_dicc[nutrient['nutrient_id']] = float(measure['value'])
                            else:
                                nut_dicc[nutrient['nutrient_id']] = float(nutrient['value'])
                        except:
                            print(ndbno)
                            sys.exit(1)
    #            ans = {'NDB_No': ndbno, 'USDA Name': name}
            ans = {'NDB_No': ndbno, 'USDA Name': name}
            for key, value in nut_dicc.items():
                ans[name_dicc[key]] = value
            arr += [ans]
            time.sleep(1)
    nutri_df = pd.DataFrame(arr)
    nutri_df = nutri_df[['NDB_No', 'USDA Name', 'Energy (Kcal)', 'Total Sugar(g)', 'Total Lipid (g)', 'Protein(g)',
             'Sodium(mg)', 'Fiber(g)', 'Calcium(mg)', 'Iron (mg)', ]]
    #print(nutri_df.info())
    nutri_df.drop_duplicates(inplace = True)
   
         
    isempty = nutri_df.empty
     
    if isempty == False:
        return nutri_df
    else:
         print("Unable to fetch Data for nutrition from USDA")
        
        
        
        
#            for key, value in nut_dicc.items():
                
                
                
                
                
#                ans[name_dicc[key]] = value
#            arr += [ans]
#            time.sleep(1)
#    nutri_data = pd.DataFrame(arr)
#    nutri_data = nutri_data[['NDB_No', 'USDA Name', 'Energy (Kcal)', 'Total Sugar(g)', 'Total Lipid (g)', 'Water (g)', 'Protein(g)',
#             'Sodium(mg)', 'Fiber(g)', 'Calcium(mg)', 'Iron (mg)', ]]
#    print(nutri_data)
    
    
    
