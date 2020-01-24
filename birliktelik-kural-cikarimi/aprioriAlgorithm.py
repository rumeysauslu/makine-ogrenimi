


import pandas as pd

# buradaki verilerin kolon basligi yok. Dolayisiyla ilk satirin kolon basligi olup olmadigini vermek icin header kullaniyoruz.
veriler = pd.read_csv('sepet.csv', header = None)

t = []
for i in range(0, 7500):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])

# ozel kutuphane kullanimi
from apyori import apriori

# apriori varsayilan destek degeri = %10, guven degeri ise = %80
# analiz edilen alisverislerin %1 inde bu urunler birlikte geciyor, % 50 oraninda ayni alisveriste ayni anda aliniyorlar.
rules = apriori(t, min_support=0.03, min_confidence=0.8, min_lift=2.1, min_length=3)
print(list(rules))
