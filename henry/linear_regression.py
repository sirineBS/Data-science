# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:12:27 2018

@author: martinhe
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    __spec__=None

    df_train = pd.read_csv(r"..\data\train.csv")
    df_test = pd.read_csv(r"..\data\test.csv")
    y = df_train['SalePrice']
    #drop na columns
    #be strict with test columns
    df_test.dropna(axis=1, inplace=True)
    
    df_train = df_train.loc[:,df_test.columns]
    #for now only take numeric features
    #df_test = pd.get_dummies(df_test)
    #df_train = pd.get_dummies(df_train)
    df_train_num = df_train.select_dtypes(exclude='object')
    
    #split   
    ss = ShuffleSplit(test_size=0.1)
    train_ix, test_ix = next(ss.split(df_train_num))
    Xtrain_raw = df_train_num.iloc[train_ix,:]
    Xtest_raw = df_train_num.iloc[test_ix,:]
    ytrain_raw = y.iloc[train_ix]
    ytest_raw = y.iloc[test_ix]
    
    #standardize
    y_scaler = StandardScaler()
    x_scaler = StandardScaler()
    Xtrain = x_scaler.fit_transform(Xtrain_raw)
    Xtest = x_scaler.transform(Xtest_raw)
    ytrain = y_scaler.fit_transform(ytrain_raw.values.reshape((-1,1)))
    ytest = ytest_raw
    #ytest = y_scaler.transform(ytest_raw.values.reshape((-1,1)))
    
    
    model = ElasticNet(max_iter=10000)
    #modle = LassoCV()
    
    params = {'alpha': np.logspace(-2,2,3),
              'l1_ratio': np.linspace(0,1,3)}
    
    grid = GridSearchCV(estimator=model, cv=5, param_grid=params, return_train_score=False,
                        n_jobs=4)
    
    grid.fit(Xtrain, ytrain)
    ypred = grid.predict(Xtest)
    
    ypred = y_scaler.inverse_transform(ypred)
    
    result = np.sqrt(mean_squared_error(ytest,ypred))
    
    #one hot stuff
    #
    print(result)
