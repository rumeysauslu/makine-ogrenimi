#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:25:51 2020

@author: rumeysa
"""

# her gösterimdeki tiklanma oranlarini veren 10 farkli reklamin 10.000 gosterimi
# zamana bagli olarak ilerliyor.
# gecmise bakarak o gun icin bir reklam onerebiliyoruz.
# takviyeli ogrenmenin amaci; gecmis verilere bakarak tecrube edinip o tecrubeleri kullanmak

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N = 10000
d = 10
toplam = 0
secilenler = []

for n in range(0, N):
    ad = random.randrange(d) # 10 tane ilandan birine tiklayacak, tiklanan ilani gosterir.
    secilenler.append(ad)
    # tiklanan reklam daha once tiklanmis bir reklam ise odul eklemek istiyoruz.
    odul = veriler.values[n, ad] # verilerdeki n.satir = 1 ise odul 1 oluyor.
    # eger tiklanani bulduysak toplam odul 1 artiyor, bulamadiysak 0 artiyor.
    toplam = toplam + odul

plt.hist(secilenler)
plt.show( ) 
    
    
    
    
    
    
    
    
