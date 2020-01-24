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

veriler = pd.read_csv('veriler.csv')

# 2.3. Encoder Kategorik -> Sayısal

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features='all')
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


cins = veriler.iloc[:,-1:].values
print(cins)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cins[:,0] = le.fit_transform(cins[:,0])
print(cins)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features='all')
cins = ohe.fit_transform(cins).toarray()
print(cins)

# Numpy Dizileri DataFrame Dönüşümü

sonuc = pd.DataFrame(data = ulke,  index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ['boy', 'kilo', 'yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)


sonuc3 = pd.DataFrame(data = cins[:,0:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)

# DataFrame Birleştirme İşlemi

s = pd.concat([sonuc,sonuc2], axis = 1)
print(s)

s2 = pd.concat([s, sonuc3], axis = 1)
print(s2)

# Verilerin Eğitim ve Test İçin Bölünmesi

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size = 0.33, random_state = 0)

# Verilerin Ölçeklendirilmesi

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


# Çoklu Regresyon

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


# Boy Tahmini İçin Çoklu Regresyon
boy = s2.iloc[:,3:4].values
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol, sag], axis = 1)
print(veri)

x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size = 0.33, random_state = 0)

reg2 = LinearRegression()
reg2.fit(x_train,y_train)

y_pred = reg2.predict(x_test)
