#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 20:22:25 2019

@author: rumeysa
"""

# 1. Kütüphaneler

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  

# 2. Veri Önişleme
# 2.1. Veri Yükleme

veriler = pd.read_csv('odev_tenis.csv')

# 2.3. Encoder Kategorik -> Sayısal
    
hava = veriler.iloc[:,0:1].values
print(hava)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features='all')
hava = ohe.fit_transform(hava).toarray()
print(hava)

''' overcast: 0
    rainy: 1
    sunny: 2
'''

ruzgar = veriler.iloc[:,3:4].values
print(ruzgar)

ruzgar = ohe.fit_transform(ruzgar).toarray()
print(ruzgar)

oyun = veriler.iloc[:,4:].values
print(oyun)

oyun = ohe.fit_transform(oyun).toarray()
print(oyun)

# Numpy Dizileri DataFrame Dönüşümü

sonuc = pd.DataFrame(data = hava,  index = range(14), columns = ['bulutlu','yagmurlu','gunesli'])
print(sonuc)

sonuc2 = pd.DataFrame(data = ruzgar[:,0:1], index = range(14), columns = ['ruzgar durumu'])
print(sonuc2)

sonuc3 = pd.DataFrame(data = oyun[:, 0:1], index = range(14), columns = ['oyun durumu'])
print(sonuc3)


# DataFrame Birleştirme İşlemi

sicaklik = veriler.iloc[:,1:2]
print(sicaklik)

nem = veriler.iloc[:,2:3]
print(nem)

s = pd.concat([sonuc,nem], axis = 1)
print(s)

s2 = pd.concat([s, sonuc2], axis = 1)
print(s2)

s3 = pd.concat([s2, sonuc3], axis = 1)
print(s3)

# Verilerin Eğitim ve Test İçin Bölünmesi

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s3, sicaklik, test_size = 0.33, random_state = 0)


# Çoklu Regresyon Modelleme

from sklearn.linear_model import LinearRegression
regresyon = LinearRegression()

regresyon.fit(x_train, y_train)

tahmin = regresyon.predict(x_test)

# OLS İstatistik Raporlama p-value

import statsmodels.api as sm
X = np.append(arr = np.ones((14, 1)).astype(int), values = s3, axis = 1)
X_l = s3.iloc[:, [0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sicaklik, exog = X_l)
r = r_ols.fit()
print(r.summary())

X_l = s3.iloc[:, [0,1,2,3,4]].values
r_ols = sm.OLS(endog = sicaklik, exog = X_l)
r = r_ols.fit()
print(r.summary())

X_l = s3.iloc[:, [0,1,2,3]].values
r_ols = sm.OLS(endog = sicaklik, exog = X_l)
r = r_ols.fit()
print(r.summary())


x_train, x_test, y_train, y_test = train_test_split(X_l, sicaklik, test_size = 0.33, random_state = 0)

reg = LinearRegression()
reg.fit(x_train, y_train)
tahmin1 = reg.predict(x_test)




