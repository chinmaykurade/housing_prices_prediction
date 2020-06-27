# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:26:09 2020

@author: chinm
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#%%
def clean(path):
    
    df = pd.read_csv(path)
    
    #%% Separating Categorical columns as ordinal or nominal
    
    cols = df.columns.tolist()
    
    cols.remove('Id')
    
    col_num = [col for col in cols if df[col].dtype=='float64' or \
               df[col].dtype=='int64']
    
    col_obj = list(set(cols)-set(col_num))
    
    # We classify them by going through data description
        
    col_nom = ['MSZoning','Street','Alley','LotShape','LandContour','LotConfig',\
               'LandSlope','Neighborhood','Condition1','Condition2','BldgType',\
               'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',\
               'MasVnrType','Foundation','Heating','Electrical','GarageType',\
               'MiscFeature','SaleType','SaleCondition','Utilities','Fence']
    
    col_ord = [col for col in col_obj if col not in col_nom]
    
    # Further classify nominal as having low or high cardinality
    low_cardinality_nom = [col for col in col_nom if df[col].nunique()<10]
    high_cardinality_nom = list(set(col_nom)-set(low_cardinality_nom))
    
    # OneHotEncode nominal columns with cardinality less than 10 and rest will be
    # label encoded
    col_OHE = low_cardinality_nom
    col_LE = col_ord+high_cardinality_nom
    
    # len(col_num+col_OHE+col_LE)
    
    #%% One Hot encoding for nominal variables
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    df_OHE = pd.DataFrame(OH_encoder.fit_transform(df[col_OHE]))
    df_OHE.index = df.index
    df_OHE.columns = OH_encoder.get_feature_names(df[col_OHE].columns.tolist())
    
    # col_OHE_final = df_OHE.columns.tolist()
    # drop_col_OHE = []
    
    # for col in col_OHE:
    #     col_start = [c for c in col_OHE_final if c.split('_')[0]==col]
    #     drop_col_OHE.append(col_start[0])
        
    # df_OHE.drop(columns=drop_col_OHE,axis=1,inplace=True)
    
    #%% Label Encoding for ordinal variables
    
    label_encoder = LabelEncoder()
    df_LE = df[col_LE].copy()
    for col in col_LE:
        df_LE[col] = label_encoder.fit_transform(df[col])
    
    
    #%% Combining all the columns and saving final dataset
    df_num = df.drop(columns=col_obj,axis=1)
    
    df_final = pd.concat([df_num,df_OHE,df_LE],axis=1)
    
    return df_final

#%%
path_train = "dataset/train_preprocessed.csv"
path_test = "dataset/test_preprocessed.csv"
#%%

df_train = clean(path_train)
df_test = clean(path_test)
    
df_test.to_csv("dataset/test_final.csv",index=False)  
df_train.to_csv("dataset/train_final.csv",index=False)

#%%

df_train.info()
df_test.info()

ncol = [col for col in df_train.columns.tolist() if col not in df_test.columns.tolist()]
ncol
