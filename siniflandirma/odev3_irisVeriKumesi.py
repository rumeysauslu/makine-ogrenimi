#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:17:35 2019

@author: rumeysa
"""

# kütüphaneler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



# veri okuma
veriler = pd.read_excel("Iris.xls")

x = veriler.iloc[:,0:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:]  #bağımlı değişken
print (x)
print (y)

# egitim ve test bolme
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# verilerin olceklendirilmesi
# verilerin her birinin esit anlamda sonuca etki etmesi icin kullanilir.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# fit = egitme
# transform = egitildigi egitimi uygulama
# fit_transform = x_trainden ogren ve o ogrendigini uygula, x_test icin yeniden egitime girme, daha once ogrendigini uygula

# log. reg.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
print(lr_pred)
print(y_test)

cm = confusion_matrix(y_test, lr_pred)
print("lojistik regresyon cm")
print (cm)

# knn 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
print(knn_pred)

cm = confusion_matrix(y_test, knn_pred)
print("KNN cm")
print(cm)

# svm
from sklearn.svm import SVC
svm = SVC(kernel='linear') # rbf, linear, poly, sigmoid, precomputed
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
print(svm_pred)

cm = confusion_matrix(y_test, svm_pred)
print ("svm cm")
print(cm)

# naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
print(nb_pred)

cm = confusion_matrix(y_test, nb_pred)
print("naive bayes cm")
print (cm)

# decision trees
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="gini")

dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print (dt_pred)

cm = confusion_matrix(y_test, dt_pred)
print("decision tree cm")
print(cm)

# random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=5, criterion="entropy")

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(rf_pred)

cm = confusion_matrix(y_test, rf_pred)
print("random forest cm")
print(cm)


