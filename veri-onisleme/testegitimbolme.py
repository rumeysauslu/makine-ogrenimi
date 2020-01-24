#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 20:22:25 2019

@author: rumeysa
"""



import pandas as pd # veriler için kullanılan kütüphane
import numpy as np # büyük sayılar ve hesaplama için kullanılan kütüphane
import matplotlib.pyplot as plt # çizimler için kullanılan kütüphane 

# veri yükleme

veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

# veri ön işleme

boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)


class insan: 
    boy = 180
    def kosmak(self,b):
        return b + 10
    
ali = insan()
print(ali.boy)
print(ali.kosmak(9))


# eksik veriler
from sklearn.preprocessing import Imputer
#preprocessing : ön işleme
imputer = Imputer(missing_values= 'NaN', strategy = ('mean'), axis = 0)
Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas)
Yas = imputer.transform(Yas)
print(Yas)


ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


sonuc = pd.DataFrame(data = ulke,  index = range(22), columns = ['fr','tr','us']) 
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ['boy', 'kilo', 'yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)

s = pd.concat([sonuc,sonuc2], axis = 1)
print(s)

s2 = pd.concat([s, sonuc3], axis = 1)
print(s2)

''' concat: iki tane ayrı dataFrame i birleştiriyor.
axis:1 diyerek alt alta değilde satır bazında bir birleştirme sağlıyoruz.

-----------------------------------------------------------------------


SONUÇ olarak elimizdeki verilere makinenin anlayabilmesi adına bir önişleme
yaptık. Eksik verilerini, farklı veri tipleri gibi sorunları çözdük. 
Son olarak ise önişlediğimiz bu verilerle bir DataFrame oluşturduk. 
Concat ile de bu DataFrame leri birleştirerek kullanılabilecek şekildeki
verimizi elde ettik. '''


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size = 0.33, random_state = 0)

''' verilerimizi test ve eğitim olarak bölüyoruz.
eğitim olarak verilerimizin 2/3 ünü,test için 1/3 ünü kullanıcaz.(test_size ile bunu belirtiyoruz)

train_test_split : verileri ayırmak için bir yöntem

s: vereceğimiz veriler(x)
sonuc3:  s sonucunda bize vermesini beklediğimiz sonuc verilerini içeriyor.(y)

****randomluk durumu makinenin başarı yüzdesini etkileyen bir durumdur****

eğitim verilerimiz x_train ve y_train 
x de yas,kilo,boy ve ülke kullanırken; y de cinsiyet kullanıyoruz.

x_train ve y_train ile eğittiğimiz makineden x_test verisini verdiğimizde y_test sonucunu bekliyoruz.

verdiğimiz boy,kilo,yas,ülkeye göre bir cinsiyet çıkarımı öğrenimi. '''














