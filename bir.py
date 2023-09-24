
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('veri.csv')


print(data.head())


data = data.fillna(0)


plt.figure(figsize=(10, 6))
plt.plot(data['Tarih'], data['Değer'], marker='o', linestyle='-')
plt.title('Örnek Veri Analizi ve Görselleştirme')
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.grid(True)
plt.show()
