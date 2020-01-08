# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:18:15 2019

@author: sonik
"""


import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix




if __name__ == '__main__':
    
    path = "C:/Users/sonik/Documents/Advance Research/instacart_2017/step-wise-analysis/buyer_count.csv"

    df = pd.read_csv(path)

    user_id = 0

    aisles = pd.read_csv("C:/Users/sonik/Documents/Advance Research/instacart_2017/step-wise-analysis/aisles.csv")
    productsdf = pd.read_csv("C:/Users/sonik/Documents/Advance Research/instacart_2017/step-wise-analysis/products.csv")
    departments = pd.read_csv("C:/Users/sonik/Documents/Advance Research/instacart_2017/step-wise-analysis/departments.csv")

    aisles = aisles[aisles["aisle"].apply(lambda x: x != "missing" and x != "other")]
    departments = departments[departments["department"].apply(lambda x: x != "missing" and x != "other")]
    print(df)
    df.rename(columns={'count': 'purchase'},inplace = True)
    df.info()
    user_to_product_interaction_train_matrix,user_to_product_interaction_test_matrix,product_to_feature_interaction_matrix

   


def user_chart():
    user_id = int(input("Enter ID"))
    data_o = df[(df.user_id == user_id) & (df.reordered == 1)][['product_name', 'purchase', 'user_id']]
    print(data_o)
    
def user_purchase_patteren():
    user_id = int(input("Enter ID"))
    data_o = df[(df.user_id == user_id) & (df.reordered == 1)][['order_dow', 'order_hour_of_day', 'user_id']]
    plt.scatter(data_o.order_dow,data_o.order_hour_of_day)
    plt.title("User Order Placing Patteren")
    plt.xlabel("Order Day")
    plt.ylabel("Order Hour")
    plt.show()


class Recommender(object):
    
    user_list = []
    item_list = []
    productsdf = pd.DataFrame()
    
    def __init__(self):
        print("In Recommendation Analysis of data by matrix factorization")
    
    def split_data(df):
        user_to_product_train_df = df[(df['eval_set'] == "prior")][['user_id','order_id','product_id','product_name','purchase']]
        
        user_to_product_test_df = df[(df['eval_set'] == "train")][['user_id','order_id','product_id','product_name','purchase']]
      
        return user_to_product_train_df,user_to_product_test_df
    
    def deldata_frames():
        del aisles 
        del departments 
        del productsdf
        

    def get_user_list(df, user_column):
          user_list = np.sort(df[user_column].unique())
          print(user_list)
          return np.sort(df[user_column].unique())
      
    def item_list(df, item_name_column):
          item_list = productsdf[item_name_column].unique()
          print(item_list)
    
    
          return item_list  
      
    def get_feature_list(aisle_df, department_df, aisle_name_column, department_name_column):
        aisle = aisle_df[aisle_name_column]
        department = department_df[department_name_column]
        
        return pd.concat([aisle, department], ignore_index = True).unique()
    
    
    
    def id_mappings(user_list, item_list, feature_list):
  
        user_to_index_mapping = {}
        index_to_user_mapping = {}
        for user_index, user_id in enumerate(user_list):
            user_to_index_mapping[user_id] = user_index
            index_to_user_mapping[user_index] = user_id
        
        item_to_index_mapping = {}
        index_to_item_mapping = {}
        for item_index, item_id in enumerate(item_list):
            item_to_index_mapping[item_id] = item_index
            index_to_item_mapping[item_index] = item_id
        
        feature_to_index_mapping = {}
        index_to_feature_mapping = {}
        for feature_index, feature_id in enumerate(feature_list):
            feature_to_index_mapping[feature_id] = feature_index
            index_to_feature_mapping[feature_index] = feature_id
        
        
        return user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping, \
           feature_to_index_mapping, index_to_feature_mapping

        def get_interaction_matrix(df, df_column_as_row, df_column_as_col, df_column_as_value, row_indexing_map, 
                              col_indexing_map):
        
            row = df[df_column_as_row].apply(lambda x: row_indexing_map[x]).values
            col = df[df_column_as_col].apply(lambda x: col_indexing_map[x]).values
            value = df[df_column_as_value].values
            
            return coo_matrix((value, (row, col)), shape = (len(row_indexing_map), len(col_indexing_map)))
        
                  
        def get_product_feature_interaction(product_df, aisle_df, department_df, aisle_weight = 1, department_weight = 1):
           
            item_feature_df = product_df.merge(aisle_df).merge(department_df)[["product_name", "aisle", "department"]]
            
            # start indexing
            item_feature_df["product_name"] = item_feature_df["product_name"]
            item_feature_df["aisle"] = item_feature_df["aisle"]
            item_feature_df["department"] = item_feature_df["department"]
            
            # allocate aisle and department into one column as "feature"
            
            product_aisle_df = item_feature_df[["product_name", "aisle"]].rename(columns = {"aisle" : "feature"})
            product_aisle_df["feature_count"] = aisle_weight # adding weight to aisle feature
            product_department_df = item_feature_df[["product_name", "department"]].rename(columns = {"department" : "feature"})
            product_department_df["feature_count"] = department_weight # adding weight to department feature
            
            # combining aisle and department into one
            product_feature_df = pd.concat([product_aisle_df, product_department_df], ignore_index=True)
            
            # saving some memory
            del item_feature_df
            del product_aisle_df
            del product_department_df
            
            
            # grouping for summing over feature_count
            product_feature_df = product_feature_df.groupby(["product_name", "feature"], as_index = False)["feature_count"].sum()
            
            return product_feature_df
        
        def main():
            users =  Recommender.get_user_list(df,'user_id')
            items = Recommender.item_list(productsdf, "product_name")
            features = Recommender.get_feature_list(aisles, departments, "aisle", "department")
            
            
#                    item_to_index_mapping[item_id] = i
#                    index_to_item_mapping[item_index] = item_id
#                    
                
            ## generate Mappings()
            user_to_index_mapping, index_to_user_mapping, \
            item_to_index_mapping, index_to_item_mapping, \
            feature_to_index_mapping, index_to_feature_mapping = Recommender.id_mappings(users, items, features)
           
            df_user = df[['user_id','product_name','purchase']]
            df_user
            product_to_feature = Recommender.get_product_feature_interaction(product_df = productsdf, 
                                                 aisle_df = aisles, 
                                                 department_df = departments,
                                                 aisle_weight=1, 
                                                 department_weight=1)
            product_to_feature.head()
           
           
           
            user_to_product_rating_train, user_to_product_rating_test = Recommender.split_data(df)
            
            user_to_product_interaction_train_matrix = Recommender.get_interaction_matrix(user_to_product_rating_train, "user_id", 
                                                "product_name", "purchase", user_to_index_mapping, item_to_index_mapping)

            # generate user_item_interaction_matrix for test data
            user_to_product_interaction_test_matrix = Recommender.get_interaction_matrix(user_to_product_rating_test, "user_id", 
                                                                "product_name", "purchase", user_to_index_mapping, item_to_index_mapping)
            
            # generate item_to_feature interaction
            product_to_feature_interaction_matrix = Recommender.get_interaction_matrix(product_to_feature, "product_name", "feature",  "feature_count", 
                                                                    item_to_index_mapping, feature_to_index_mapping)
                          
    
            if __name__ == "__main__":
                main()
                
   


class fit_Model(object):
    
    
    
    model_without_features = LightFM(loss = "warp")
             
                       

    
    
    
    
    
        
    
    


    
   