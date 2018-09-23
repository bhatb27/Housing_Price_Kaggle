# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:13:13 2018

@author: Karthik Bhat
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

#Importing train and test data
dataset = pd.read_csv('C:\\Study\\Kaggle\\House Prices\\train.csv')
test_dataset = pd.read_csv('C:\\Study\\Kaggle\\House Prices\\test.csv')

fin_set = pd.concat([dataset,test_dataset],)

X = fin_set.describe()
X = fin_set.info()

#Missing value correction
X = fin_set.isnull().sum()
train = fin_set.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],1)
X = train.isnull().sum()

imputer_num = Imputer()
imputer_cat = Imputer(strategy="most_frequent",axis=1)
encoder = LabelEncoder()

train[['LotFrontage']] = imputer_num.fit_transform(train[['LotFrontage']])
train[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']] = train[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].fillna('NA')
train[['MasVnrType']] = train[['MasVnrType']].fillna('None')
train[['MasVnrArea']] = train[['MasVnrArea']].fillna(0)
train[['Electrical']] = train[['Electrical']].fillna(train.Electrical.mode().values[0])
train[['GarageType','GarageFinish','GarageQual','GarageCond']] = train[['GarageType','GarageFinish','GarageQual','GarageCond']].fillna('Na')
train[['GarageYrBlt']] = train[['GarageYrBlt']].fillna('Na')
train[['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF']] = train[['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF']].fillna(0)
train[['MSZoning']] = train[['MSZoning']].fillna(train.MSZoning.mode().values[0])
train[['Functional']] = train[['Functional']].fillna(train.Functional.mode().values[0])
train[['Utilities']] = train[['Utilities']].fillna(train.Utilities.mode().values[0])
train[['Exterior1st']] = train[['Exterior1st']].fillna(train.Exterior1st.mode().values[0])
train[['Exterior2nd']] = train[['Exterior2nd']].fillna(train.Exterior2nd.mode().values[0])
train[['GarageArea','GarageCars']] = train[['GarageArea','GarageCars']].fillna(0)
train[['KitchenQual']] = train[['KitchenQual']].fillna(train.KitchenQual.mode().values[0])
train[['SaleType']] = train[['SaleType']].fillna(train.SaleType.mode().values[0])

X = train.describe()

#Checking distribution of target attribute
train['SalePrice'].hist()
train['lnSalePrice'] = np.log(train['SalePrice'])
train['lnSalePrice'].hist()

train['BldgType'] = train['BldgType'].astype('category')

train2 = train[:]

for col in train2.columns:
    if train2[col].dtype == 'object':
        train2[col] = train2[col].astype('category')

X = train2.dtypes

train3 = pd.get_dummies(train2,drop_first = True)

corrl = train2.corr()

X = np.where(corrl<0.8)





















