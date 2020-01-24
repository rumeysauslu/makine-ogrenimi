#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:25:29 2020

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

# bu tablodaki sonuclara gore;
# artik n_clusters yani k degerini en uygun deger secebiliriz.
# egimlerin degisiklik gosterdigi noktalar alinabilir.
