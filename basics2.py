import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('veri.csv')

# Veriyi gözlemleme
print(data.head())

# Eksik verileri kontrol etme ve temizleme
missing_data = data.isnull().sum()
print("Eksik Veri Sayısı:\n", missing_data)

# Kategorik verileri kodlama 
data = pd.get_dummies(data, columns=['kategorik_sutun'])

# Veriyi bağımsız ve bağımlı değişkenlere ayırma
X = data.drop('hedef_sutun', axis=1)
y = data['hedef_sutun']

# Eğitim ve test veri kümelerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Makine öğrenimi modelini oluşturma 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modeli test veri kümesi üzerinde değerlendirme
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Doğruluğu:", accuracy)

# Sınıflandırma raporu oluşturma
class_report = classification_report(y_test, y_pred)
print("Sınıflandırma Raporu:\n", class_report)

# Model performansını görselleştirme 
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=X.columns)
plt.title('Özellik Önem Sıralaması')
plt.xlabel('Özellik Önem Derecesi')
plt.ylabel('Özellikler')
plt.show()
