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

# aylar: bağımsız değişken, satış: aylara bağımlı değişken 

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
# fit ile bir model oluşturuyoruz. Bu model X_train den Y_train i tahmin edecek.
lr.fit(x_train, y_train)

# Model Uygulama

# x_test verisini kullanarak tahmin sonucu oluşturuyoruz.
tahmin = lr.predict(x_test)

# Verileri Görselleştirme

x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train, y_train)

plt.plot(x_test, tahmin)

# plt ayrıntıları

plt.title("aylara göre satış")
plt.xlabel("aylar")
plt.ylabel("satışlar")






