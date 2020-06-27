# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:24:08 2020

@author: chinm
"""

import pandas as pd
from sklearn.impute import SimpleImputer
pd.set_option('max_rows', 50)
pd.set_option('max_columns', None)

#%%
def get_preprocessed(path):
    
    df = pd.read_csv(path)
    
    #%%
    # print(df.info())
    
    # print(df.describe())
    
    # print(df.isnull().sum())
    
    #%%
    # Columns of missing numerical values and missing stings
    colmiss = [col for col in df.columns if df[col].isnull().any()]
    
    colmiss_num = [col for col in colmiss if df[col].dtype==float or df[col].dtype==int]
    
    colmiss_obj = list(set(colmiss)-set(colmiss_num))
    
    #%% Dealing with missing numeric values
    print("Missing numerical values:\n",df[colmiss_num].isnull().sum())
    
    values = df[colmiss_num].median()
    df[colmiss_num] = df[colmiss_num].fillna(value=values)
    
    # df.drop(columns = colmiss_num,inplace=True,axis=1)
    
    #%% Dealing with missing categorical values
    print("\nMissing categorical values:\n",df[colmiss_obj].isnull().sum())
    
    # By foing through the dataset info, we see that missing categorical values 
    # actually refer to the given categorical item not being available
    my_imputer = SimpleImputer(strategy='constant',fill_value='NotAvailable')
    imputed_obj_cols = pd.DataFrame(my_imputer.fit_transform(df[colmiss_obj]))
    
    imputed_obj_cols.columns = df[colmiss_obj].columns
    
    df[colmiss_obj] = imputed_obj_cols
    
    return df

#%%
path_train = "dataset/train.csv"
path_test = "dataset/test.csv"

#%%

df_train = get_preprocessed(path_train)
df_test = get_preprocessed(path_test)
    
df_test.to_csv("dataset/test_preprocessed.csv",index=False)  
df_train.to_csv("dataset/train_preprocessed.csv",index=False)

#%%

