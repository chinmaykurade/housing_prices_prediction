# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:12:42 2020

@author: chinm
"""

import pandas as pd
pd.set_option('max_rows', None)

#%%

# Read the train and test datasets
df_train = pd.read_csv("dataset/train_final.csv")
df_test = pd.read_csv("dataset/test_final.csv")

cols_train = df_train.columns.tolist()
cols_test = df_test.columns.tolist()

# Train the model with columns that exist both in train and test set
cols_to_train = [col for col in cols_train if col in cols_test]
X_test_all = df_test[cols_to_train]

cols_to_train.remove('Id')
cols_all = cols_to_train+['SalePrice']
X_train_all = df_train[cols_all]


#%%

print("Dataset shape : {}".format(X_train_all.shape))

X_corr = X_train_all.corr()['SalePrice'].sort_values(ascending=False)

col_poly = [col for col in X_corr.index.tolist() if abs(X_corr[col]>0.1)]
col_poly.remove('SalePrice')

## We will create degree=2 features for columns with correlation > 0.1
X_train_poly = X_train_all[col_poly]**2
X_test_poly = X_test_all[col_poly]**2

X_train_poly.columns = [col+'_2' for col in col_poly]
X_test_poly.columns = [col+'_2' for col in col_poly]

X_train_all = pd.concat([X_train_all,X_train_poly],axis=1)
X_test_all = pd.concat([X_test_all,X_test_poly],axis=1)

#%%


X_train_all.to_csv("dataset/train_poly.csv")
X_test_all.to_csv("dataset/test_poly.csv")

