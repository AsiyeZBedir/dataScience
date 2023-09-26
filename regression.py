import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('veri.csv')

# Veriyi gözlemleme
print(data.head())

# Eksik verileri kontrol etme ve temizleme
missing_data = data.isnull().sum()
print("Eksik Veri Sayısı:\n", missing_data)

# Özellik mühendisliği 
data['tarih'] = pd.to_datetime(data['tarih'])
data['yıl'] = data['tarih'].dt.year
data['ay'] = data['tarih'].dt.month
data['hafta'] = data['tarih'].dt.week

# Kategorik verileri kodlama
data = pd.get_dummies(data, columns=['kategorik_sutun'])

# Veriyi bağımsız ve bağımlı değişkenlere ayırma
X = data.drop('hedef_sutun', axis=1)
y = data['hedef_sutun']

# Eğitim ve test veri kümelerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Makine öğrenimi modelini oluşturma 
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Modeli test veri kümesi üzerinde değerlendirme
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Ortalama Kare Hata (MSE):", mse)
print("R-Kare (R^2) Değeri:", r2)

# Model performansını görselleştirme 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahminler')
plt.title('Tahminler vs. Gerçek Değerler')
plt.show()
