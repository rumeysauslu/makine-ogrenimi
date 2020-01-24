#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 20:22:25 2019

@author: rumeysa
"""

# Kütüphaneler
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  
import statsmodels.api as sm


# Veri Yükleme
veriler = pd.read_csv('maaslar_yeni.csv')
print(veriler)

x = veriler.iloc[:,2:3]
y = veriler.iloc[:, 5:]

X = x.values
Y = y.values

# Lineer Reg.
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

r_ols = sm.OLS(lin_reg.predict(X), X)
# X in tahmin değeri ile X i karşılaştır.
print(r_ols.fit().summary())

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin2 = LinearRegression()
lin2.fit(x_poly, Y)

plt.title("polinomal")
plt.scatter(X, Y)
plt.plot(X, lin2.predict(x_poly))
plt.show()

# SVR
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x_olcekli = std.fit_transform(X)
std2 = StandardScaler()
y_olcekli = std.fit_transform(Y)

from sklearn.svm import SVR
sv = SVR(kernel = 'rbf')
sv.fit(x_olcekli, y_olcekli)

plt.title("destek vektör")
plt.scatter(x_olcekli, y_olcekli)
plt.plot(x_olcekli,sv.predict(x_olcekli))
plt.show()

# Karar Ağacı
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=0)
dt.fit(X,Y)

plt.title("karar ağacı")
plt.scatter(X,Y)
plt.plot(X, dt.predict(X))
plt.show()

# Rassal Ağaç
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(X,Y)

plt.title("Rassal Ağaç")
plt.scatter(X,Y)
plt.plot(X, rf.predict(X))
plt.show()

# başarı karşılaştırması
from sklearn.metrics import r2_score
print("------------------")
print("doğrusal reg")
print(r2_score(Y, lin_reg.predict(X)))

print("polinomal reg")
print(r2_score(Y, lin2.predict(x_poly)))

print("svr reg")
print(r2_score(y_olcekli, sv.predict(x_olcekli)))

print("karar ağacı reg")
print(r2_score(Y, dt.predict(X)))

print("rassal ağaç reg")
print(r2_score(Y, rf.predict(X)))
