#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:57:13 2019

@author: rumeysa
"""
import pandas as pd

veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

yas = veriler.iloc[:,1:4].values
print(yas)

from sklearn.preprocessing import Imputer
impute = Imputer(missing_values='NaN', strategy='mean', axis=0)

y = impute.fit_transform(yas)
print(y)

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
snc = ohe.fit_transform(ulke).toarray()
print(snc)