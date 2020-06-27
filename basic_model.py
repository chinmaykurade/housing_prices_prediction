# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:08:59 2020

@author: chinm
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
pd.set_option('max_rows', 50)
pd.set_option('max_columns', None)


df = pd.read_csv("dataset/train_final.csv")

#%%
cols = list(df.columns)

cols.remove('Id')

col_num = [col for col in cols if df[col].dtype=='float64' or \
           df[col].dtype=='int64']

col_obj = list(set(cols)-set(col_num))

# Columns of missing numerical values and missing stings
colmiss = [col for col in cols if df[col].isnull().any()]

colmiss_num = [col for col in colmiss if df[col].dtype==float or df[col].dtype==int]

colmiss_obj = list(set(colmiss)-set(colmiss_num))

#%%
cols_all = list(set(col_num)-set(colmiss_num))

Y_all = df['SalePrice']
X_all = df[cols_all].drop(columns='SalePrice')

X_train, X_valid, Y_train, Y_valid = train_test_split(X_all, Y_all,\
                                train_size=0.8, test_size=0.2,random_state=4)

def get_mse_valid(model,X_train, X_valid, Y_train, Y_valid):
    model.fit(X_train, Y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(Y_valid, preds)

model = RandomForestRegressor(n_estimators=200, random_state=4)
print(get_mse_valid(model,X_train, X_valid, Y_train, Y_valid))