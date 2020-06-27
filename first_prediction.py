# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:31:59 2020

@author: chinm
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
pd.set_option('max_rows', 50)
pd.set_option('max_columns', None)


df_train = pd.read_csv("dataset/train_final.csv")
df_test = pd.read_csv("dataset/test_final.csv")

cols_train = df_train.columns.tolist()
cols_test = df_test.columns.tolist()

# cols_to_train1 = list(set(cols_train) and set(cols_test))
cols_to_train = [col for col in cols_train if col in cols_test]
cols_to_train.remove('Id')

len(cols_train)
len(cols_test)
len(cols_to_train)

X_train_i = df_train[cols_to_train]
Y_train = df_train['SalePrice']
X_test_i = df_test[cols_to_train]
X_Id = df_test['Id']

#%%

# model = RandomForestRegressor(n_estimators=400,random_state=4)
model = LogisticRegression(solver='sag',verbose=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_i)
X_test = scaler.fit_transform(X_test_i)


model.fit(X_train, Y_train)
preds_train = model.predict(X_train)
mean_absolute_error(Y_train, preds_train)
preds_test = model.predict(X_test)


#%%

result = pd.DataFrame({
    'Id':X_Id,
    'SalePrice':preds_test
})

result.to_csv("dataset/result_2.csv",index=False)
