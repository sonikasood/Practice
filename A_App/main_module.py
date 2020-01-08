# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:32:31 2019

@author: sonik
"""

import Recommender
import sys
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score
import time
import user_recommendation
import nutri
from progress.bar import Bar
#links
#https://heartbeat.fritz.ai/recommender-systems-with-python-part-ii-collaborative-filtering-k-nearest-neighbors-algorithm-c8dcd5fd89b2



def loading(load):
    loading = load  # a simple var to keep the loading status
    loading_speed = 4  # number of characters to print out per second
    loading_string = "." * 6  # characters to print out one by one (6 dots in this example)
    while loading:
    #  track both the current character and its index for easier backtracking later
        for index, char in enumerate(loading_string):
        # you can check your loading status here
        # if the loading is done set `loading` to false and break
            sys.stdout.write(char)  # write the next char to STDOUT
            sys.stdout.flush()  # flush the output
            time.sleep(1.0 / loading_speed)  # wait to match our speed
            index += 1  # lists are zero indexed, we need to increase by one for the accurate count
    # backtrack the written characters, overwrite them with space, backtrack again:
        
def text(text_to_print,num_of_dots,num_of_loops):
    from time import sleep
    
    import sys
    shell = sys.stdout.shell
    shell.write(text_to_print,'stdout')
    dotes = int(num_of_dots) * '.'
    for last in range(0,num_of_loops):
        for dot in dotes:
            print('.')
            sleep(0.1)
        for dot in dotes:
            print('\x08')
            sleep(0.1)


def Model_fit_part1(train,test):
    # initialising model with warp loss function
   
    model_without_features = LightFM(loss = "warp")
    start = time.time()
#===================

    model_without_features.fit(train,
              user_features=None, 
              item_features=None, 
              sample_weight=None, 
              epochs=1, 
              num_threads=4,
              verbose=False)
    
    #===================
    end = time.time()
    print("time taken = {0:.{1}f} seconds".format(end - start, 2))
    
    print("checking accuracy and results with test data")
    # auc metric score (ranging from 0 to 1)

    start = time.time()
    #===================
    
    auc_without_features = auc_score(model = model_without_features, 
                            test_interactions = test,
                            num_threads = 4, check_intersections = False)
    print(auc_without_features)
    
    #===================
    end = time.time()
    print("accurcay model time taken = {0:.{1}f} seconds".format(end - start, 2))



def Model_fit_part2(train,prod_features,test):
 # initialising model with warp loss function
    from lightfm import LightFM
    
    from lightfm.evaluation import auc_score
    model_with_features = LightFM(loss = "warp")
    
    # fitting the model with hybrid collaborative filtering + content based (product + features)
    start = time.time()
    #===================
    
    
    model_with_features.fit(train,
              user_features=None, 
              item_features=prod_features, 
              sample_weight=None, 
              epochs=1, 
              num_threads=4,
              verbose=False)
    auc_with_features = auc_score(model = model_with_features, 
                            test_interactions = test,
                            train_interactions = train, 
                            item_features = prod_features,
                            num_threads = 4, check_intersections=False)
    #===================
    end = time.time()
    print("time taken = {0:.{1}f} seconds".format(end - start, 2))
    print("average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_with_features.mean(), 2))

def combined_train_test(train, test):
    """
    
    test set is the more recent rating/number_of_order of users.
    train set is the previous rating/number_of_order of users.
    non-zero value in the test set will replace the elements in 
    the train set matrices

    """
    # initialising train dict
    train_dict = {}
    for train_row, train_col, train_data in zip(train.row, train.col, train.data):
        train_dict[(train_row, train_col)] = train_data
        
    # replacing with the test set
    
    for test_row, test_col, test_data in zip(test.row, test.col, test.data):
        train_dict[(test_row, test_col)] = max(test_data, train_dict.get((test_row, test_col), 0))
        
    
    # converting to the row
    row_element = []
    col_element = []
    data_element = []
    for row, col in train_dict:
        row_element.append(row)
        col_element.append(col)
        data_element.append(train_dict[(row, col)])
        
    # converting to np array
    
    row_element = np.array(row_element)
    col_element = np.array(col_element)
    data_element = np.array(data_element)
    
    return coo_matrix((data_element, (row_element, col_element)), shape = (train.shape[0], train.shape[1]))


    
    
def start_recoomendation(df,aisles,departments,productsdf):
    
    
    
    
    ## create matrix factorization of data 
    user_to_product_interaction_train_matrix,user_to_product_interaction_test_matrix,product_to_feature_interaction_matrix,items,user_to_index_mapping = Recommender.func_cal(df,aisles,departments,productsdf)
    
    train = user_to_product_interaction_train_matrix
    test =  user_to_product_interaction_test_matrix
    prod_features = product_to_feature_interaction_matrix
    
    print(items)
   # Model_fit_part1(train,test)
    
    print("################# printing results from complete collaborative filtering ###############")
          
 #   Model_fit_part2(train,prod_features,test)     
          
          
    user_to_product_interaction = combined_train_test(train, 
                                                 test)

    print(user_to_product_interaction.shape)

    
    final_model = LightFM(loss = "warp")

# fitting to combined dataset with pure collaborative filtering result

    start = time.time()
    #===================
    
    final_model.fit(user_to_product_interaction,
              user_features=None, 
              item_features=None, 
              sample_weight=None, 
              epochs=1, 
              num_threads=4,
              verbose=False)
    
    
    #===================
    end = time.time()
    print("time taken = {0:.{1}f} seconds".format(end - start, 2))
    
    #print("average AUC without adding item-feature interaction = {0:.{1}f}".format(accuracy_final.mean(), 2))

    recom = user_recommendation.recommendation_sampling(final_model,items,user_to_product_interaction,user_to_index_mapping)
    
    recom.recommendation_for_user(1)
    
    
def user_chart(df):
    user_id = int(input("Enter ID"))
    data_o = df[(df.user_id == user_id) & (df.reordered == 1)][['product_name', 'purchase', 'user_id']]
    # sorting by first name 
    
    data_o.sort_values("purchase", axis = 0, ascending = False, 
                 inplace = True, na_position ='first') 
    data_o.drop_duplicates(inplace = True) 
    print(data_o)
   

def user_nutrichart(df):
    user_id = int(input("Enter ID"))
    data_o = df[(df.user_id == user_id)]
    # sorting by first name 
   
    data_o.sort_values("purchase", axis = 0, ascending = False, 
                 inplace = True, na_position ='first') 
    
    data_o.drop_duplicates(inplace = True)
# dropping ALL duplicte values 
   
#    data_o = data_o.drop_duplicates(inplace = True) 
  
   
    
    data_o = data_o.nlargest(10, 'purchase')
    
    
    
    nutri.nutrition(data_o)
    
   
   
    
    #return data_o
#
#    print("No data with this id")

    
def user_purchase_patteren(df):
    user_id = int(input("Enter ID"))
    data_o = df[(df.user_id == user_id) & (df.reordered == 1)][['order_dow', 'order_hour_of_day', 'user_id']]
    plt.scatter(data_o.order_dow,data_o.order_hour_of_day)
    plt.title("User Order Placing Patteren")
    plt.xlabel("Order Day")
    plt.ylabel("Order Hour")
    plt.show()
    




if __name__ == '__main__':
    
   
    


        # Do some work
        
   
        
    print("loading data from files")
    
    
    
    # Do some work
    
    path = "C:/Users/sonik/Documents/Advance Research/instacart_2017/step-wise-analysis/buyer_count.csv"
    
    df = pd.read_csv(path)
    df.drop_duplicates(inplace = True) 
    
    user_id = 0
    
    
    aisles = pd.read_csv("C:/Users/sonik/Documents/Advance Research/instacart_2017/step-wise-analysis/aisles.csv")
    productsdf = pd.read_csv("C:/Users/sonik/Documents/Advance Research/instacart_2017/step-wise-analysis/products.csv")
    departments = pd.read_csv("C:/Users/sonik/Documents/Advance Research/instacart_2017/step-wise-analysis/departments.csv")

   
    print("Complete Data Loaded")
    

    aisles = aisles[aisles["aisle"].apply(lambda x: x != "missing" and x != "other")]
    departments = departments[departments["department"].apply(lambda x: x != "missing" and x != "other")]
   
    df.rename(columns={'count': 'purchase'},inplace = True)
    
   
#   
    df.drop(df.columns[[0, 1, 2]], axis=1,inplace=True)
    
    
    
    
    
   
    
   
    
    print("hi You have Entered the Python Instacart Market Basket Analyser")
    
    
    ans = True
    while ans:
        print("Main Menu" + "Please choose your option")
        print("""
        1.Give user recommendations
        2.Analyse user Purchase History
        3.Items Frequently bought
        4.Do nutrional Analysis
        5.Exit/Quit
       
        """)
        ans=input("What would you like to do? ")
        if ans=="1":
         start_recoomendation(df,aisles,departments,productsdf)
        elif ans=="2":
          print("User Purchase History along with time he does reoerders")  
          user_purchase_patteren(df)
        elif ans=="3":
          print("User Frequently Bought Items are:") 
          user_chart(df)
        elif ans=="4":
            
            nutri_data = df[['user_id','product_id','purchase','product_name','aisle_id','department_id','reordered']]
    
            user_nutrichart(nutri_data)
          
        elif ans=="5":
          print("\n Goodbye") 
          ans = None
        else:
           print("\n Not Valid Choice Try again")


                       