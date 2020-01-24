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

veriler = pd.read_csv('maaslar.csv')
# bağlantı kurmak istediğim yer eğitim seviyesi ve maaş

x = veriler.iloc[:, 1:2].values
y = veriler.iloc[:, 2:].values

# lineer regresyon
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y)
plt.plot(x,lin_reg.predict(x))
plt.show()

# Polinomal Regresyon
from sklearn.preprocessing import PolynomialFeatures
# polynomialFeatures istediğimiz veriyi polinomal olarak ifade etmemizi sağlıyor.
poly_reg = PolynomialFeatures(degree=2)
# 2. dereceden bir obje oluşturduk.
x_poly = poly_reg.fit_transform(x)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
# x_poly den y yi öğren, modelle.

plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg2.predict(x_poly), color = 'blue')
plt.show()

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg2.predict(x_poly), color = 'blue')
plt.show()


# tahminler

print(lin_reg.predict(np.array([11]).reshape(1,-1)))
