import pandas as pd
import matplotlib.pyplot as plt
import requests

url = "https://raw.githubusercontent.com/openai/gpt-3.5-turbo/master/data/sample_data.csv"
response = requests.get(url)


df = pd.read_csv(url)


print(df.head())

# Bir sütunun histogramını çizme
plt.hist(df['Age'], bins=20)
plt.xlabel('Yaş')
plt.ylabel('Frekans')
plt.title('Yaş Dağılımı')
plt.show()
