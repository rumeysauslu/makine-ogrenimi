#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:35:23 2020

@author: rumeysa
"""
import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv')

import re # regular expression library
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer() # kelimelerin koklerini, govdelerini bulacagiz

# Stop Word (anlamsiz kelimeler is, the, ve, ama vb.)
nltk.download('stopwords')
from nltk.corpus import stopwords

derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ', yorumlar['Review'][i])
    yorum = yorum.lower() # kucuk harfe cevir
    yorum = yorum.split() # kelime kelime ayir
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] # kelimenin kokunu bul, liste olustur
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
# En Cok Kullanilan Kelimeler
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000) # en cok kullanilan 2000 kelimeyi aldik, her yorumda bu var mi yok mu bakiyoruz.
X = cv.fit_transform(derlem).toarray() # bagimsiz degisken
y = yorumlar.iloc[:,1].values # bagimli degisken

# Makine Ogrenimi - Naive Bayes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
























