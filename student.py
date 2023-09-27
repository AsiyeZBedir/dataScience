import pandas as pd
import matplotlib.pyplot as plt

data = {'Öğrenci Adı': ['Alice', 'Bob', 'Charlie', 'David'],
        'Notlar': [85, 92, 78, 88]}

df = pd.DataFrame(data)

ortalama_not = df['Notlar'].mean()

print("Notların Ortalaması:", ortalama_not)

# Notları çizdirme
plt.bar(df['Öğrenci Adı'], df['Notlar'])
plt.xlabel('Öğrenci Adı')
plt.ylabel('Notlar')
plt.title('Öğrenci Notları')
plt.show()

# Veriyi bir CSV dosyasına kaydetme
df.to_csv('ogrenci_notlari.csv', index=False)


