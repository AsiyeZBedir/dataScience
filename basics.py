
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('veri.csv')


print(data.head())
print(data.info())


summary_stats = data.describe()
print(summary_stats)

# Veriyi görselleştirme
# Örnek bir histogram çizme
plt.hist(data['sütun_adı'], bins=20)
plt.xlabel('X ekseni etiketi')
plt.ylabel('Y ekseni etiketi')
plt.title('Veri Dağılımı')
plt.show()

# Örnek bir sıcaklık haritası çizme
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()
