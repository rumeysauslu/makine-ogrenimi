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

x = veriler.iloc[:, 1:4].values # bağımsız değ.
y = veriler.iloc[:, 4: ].values # bağımlı değ.
# Verilerin Eğitim ve Test İçin Bölünmesi

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

# Verilerin Ölçeklendirilmesi

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# fit = eğitme
# transform = öğrendiği eğitimi uygulama
# x_train den öğren ve uygula ama x_test için yeniden öğrenme, daha önce öğrendiğinle uygula.

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

# confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('LOJISTIK')
print(cm)


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train, y_train)
y_knn_pred = knn.predict(X_test)


cm_knn = confusion_matrix(y_test, y_knn_pred)
print('KNN')
print(cm_knn)
  

#SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
# X_train den (x_trainin standardscaler edilmiş hali, y_traini öğren)

y_svm_pred = svm.predict(X_test)

cm_svm = confusion_matrix(y_test, y_svm_pred)
print('SVM')
print(cm_svm)

#Naif Bayes 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_gnb_pred = gnb.predict(X_test)

cm_gnb = confusion_matrix(y_test, y_gnb_pred)
print('gnb')
print(cm_gnb)






