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


# Veri Yükleme
veriler = pd.read_csv('maaslar.csv')


# Data Frame Dilimleme(slice)
x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]


# NumPY Array Dönüşümü
X = x.values
Y = y.values


# Lineer Regresyon
# Doğrusal Model Oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


# Polinomal Regresyon
# Doğrusal Olmayan(nonlinear model) Oluşturma
# 2. Dereceden Polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, Y)


# 4. Dereceden Polinom
poly_reg3 = PolynomialFeatures(degree=4)
x_poly2 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly2, Y)


# Görselleştirme
plt.scatter(X,Y)
plt.plot(X,lin_reg.predict(X))
plt.show()

plt.scatter(X,Y, color = 'red')
plt.plot(x,lin_reg2.predict(x_poly), color = 'blue')
plt.show()

plt.scatter(X,Y, color = 'red')
plt.plot(x,lin_reg3.predict(x_poly2), color = 'blue')
plt.show()


# Tahminler
print(lin_reg.predict(np.array([11]).reshape(1,-1)))


# Verilerin Ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_olcekli = sc.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

# SVR
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli, y_olcekli)

# Görsel
plt.scatter(x_olcekli,y_olcekli,color = 'red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color = 'blue')
plt.show()


# Karar Ağacı
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y)
plt.plot(X, r_dt.predict(X))
plt.show()

print(r_dt.predict(np.array([6.6]).reshape(1,-1)))

# Rassal Ağaçlar (Random Forest)
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
# estimators : kaç tane decision tree çizileceği
rf_reg.fit(X,Y)
# x bilgisinden y bilgisini öğren.
print(rf_reg.predict(np.array([6.5]).reshape(1,-1)))

plt.scatter(X,Y, color = 'red')
plt.plot(x, rf_reg.predict(X))













