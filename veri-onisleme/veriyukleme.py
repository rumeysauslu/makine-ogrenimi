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

veriler = pd.read_csv('veriler.csv') 
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

l = [1,2,3] 



