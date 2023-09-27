import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

url = "https://raw.githubusercontent.com/openai/gpt-3.5-turbo/master/data/sample_data.csv"
df = pd.read_csv(url)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Doğruluğu:", accuracy)

class_report = classification_report(y_test, y_pred)
print("Sınıflandırma Raporu:\n", class_report)

model_filename = "data_science_model.pkl"
joblib.dump(clf, model_filename)

