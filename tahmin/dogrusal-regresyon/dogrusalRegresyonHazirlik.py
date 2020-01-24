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

veriler = pd.read_csv('satislar.csv')
print(veriler)

# 2.2. Veri Önişleme
# Verileri Ayırma

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)


# Verilerin Eğitim ve Test İçin Bölünmesi

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size = 0.33, random_state = 0)

# Verilerin Ölçeklendirilmesi

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)


# Basit Doğrusal Regresyon Model İnşası

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

# Model Uygulama

tahmin = lr.predict(x_test)











