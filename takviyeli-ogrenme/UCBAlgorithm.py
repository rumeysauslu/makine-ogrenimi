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

# gecmis bilgilere gore ogrenen, rastgele secmeyen, eski bilgileri aklinda tutan, sonunda da bu bilgilerden faydalanan algoritma

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10 
# Ri(n)
oduller = [0] * d # 10 elemanli, elemani 0 olan bir dizi, butun ilanlarin odul degeri basta 0
toplam = 0 # toplam odul
# Ni(n)
tiklamalar = [0] * d # o ana kadarki tiklamalar
secilenler = []

for n in range(0,N): # her bir tiklama olayi
    ad = 0 # secilen ilan
    max_ucb = 0
    # bir ad secip ona odul donup donmedigini gostermeliyiz.
    # hangi ad i sececegim rastgele olmamali
    # en yuksek ucb degerine sahip olani almaliyiz.
    # bu yuzden ucb degerlerini hesaplamaliyiz.
    for i in range(0,d): # her bir ilan icin hangisine tiklayacagimi bulmaya yarayan bir dongu
        # bu dongunun amaci, 10 ilana da bak en yuksek ucb yi al.
        if(tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2 * math.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb 
            ad = i 
    
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n, ad]
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul

print('toplam odul: ')
print(toplam)

plt.hist(secilenler)
plt.show()
