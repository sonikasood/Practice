# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:32:31 2019

@author: sonik
"""


import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score


#links
#https://heartbeat.fritz.ai/recommender-systems-with-python-part-ii-collaborative-filtering-k-nearest-neighbors-algorithm-c8dcd5fd89b2

class recommendation_sampling:
    
     def __init__(self, model, items, user_to_product_interaction_matrix,
              user_to_index_mapping):
        
        self.user_to_product_interaction_matrix = user_to_product_interaction_matrix
        self.model = model
        self.items = items
        self.user2index_map = user_to_index_mapping
    
     def recommendation_for_user(self, user):
        
        # getting the userindex
        user=input("Enter user id to give recommendations")
        userindex = self.user2index_map.get(3, None)
        
        if userindex == None:
            return "This user Id does not exist try another one"
        
        users = [userindex]
        
        # products already bought
        
        known_positives = self.items[self.user_to_product_interaction_matrix.tocsr()[userindex].indices]
        
        # scores from model prediction
        scores = self.model.predict(user_ids = users, item_ids = np.arange(self.user_to_product_interaction_matrix.shape[1]))
        
        
        # top items
        
        top_items = self.items[np.argsort(-scores)]
        
        # printing out the result
        print("User %s" % user)
        print("     Known positives:")
        
        for x in known_positives[:4]:
            print("                  %s" % x)
            
            
        print("     Recommended:")
        
        for x in top_items[:4]:
            print("                  %s" % x)
            
            


                       