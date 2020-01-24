#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 19:57:47 2020

@author: rumeysa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:, 3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)

# k icin optimum degeri bulmaya calisiyoruz.
# k icin opt. deger deneyerek bulunur.

# amac: kmeans algoritmasinin ayni random degerle, ayni initi kullanarak farkli bir cluster(küme) sayisi belirlemesi
sonuclar = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) # inertia : wcss degerlerimiz
    # append ile bu inertia degerlerini topluyoruz ki ne kadar basarili olduğunu bulalim.

# sonuclari gorsellestirmek istersek;
plt.plot(range(1,10), sonuclar)
plt.show()

# bu tablodaki sonuclara gore;
# artik n_clusters yani k degerini en uygun deger secebiliriz.
# egimlerin degisiklik gosterdigi noktalar alinabilir.


kmeans = KMeans(n_clusters=4, init='k-means++', random_state=123)
Y_tahmin = kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin == 0, 0], X[Y_tahmin == 0, 1], s=100, c='red')
plt.scatter(X[Y_tahmin == 1, 0], X[Y_tahmin == 1, 1], s=100, c='blue')
plt.scatter(X[Y_tahmin == 2, 0], X[Y_tahmin == 2, 1], s=100, c='green')
plt.scatter(X[Y_tahmin == 3, 0], X[Y_tahmin == 3, 1], s=100, c='yellow')
plt.title('KMeans')
plt.show()


# hiyerarsik bolutleme
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)
# bunun sonucu, her bir indexin hangi kümeye ait oldugunu gosterir.
# 0,1,2 olmak üzere 3 küme var

# hem tahmin hem sonuc degerlerini alicaz.
# y tahmin degeri 0 olanlarin 0 ve 1. kolonlarindaki tahmin degerini goster.
plt.scatter(X[Y_tahmin == 0, 0], X[Y_tahmin == 0, 1], s=100, c='red')
plt.scatter(X[Y_tahmin == 1, 0], X[Y_tahmin == 1, 1], s=100, c='blue')
plt.scatter(X[Y_tahmin == 2, 0], X[Y_tahmin == 2, 1], s=100, c='green')
plt.scatter(X[Y_tahmin == 3, 0], X[Y_tahmin == 3, 1], s=100, c='yellow')
plt.title('Hiyerarsik')
plt.show()

# dendogram icin scipy dan kutuphane ekleyecegiz.
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
# x veri datasi uzerinde bir dendrogram tanimliyoruz. ward mesafesine gore hesaplayacak.
plt.show()


























