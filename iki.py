# Gerekli kütüphaneler
import pandas as pd
import numpy as np
import statsmodels.api as sm


np.random.seed(0)
X = np.random.rand(100, 2)  # İki bağımsız değişken
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.rand(100)  # Bağımlı değişken


X = sm.add_constant(X)

# Çoklu lineer regresyon modeli
model = sm.OLS(y, X).fit()


print(model.summary())


coefficients = model.params
r_squared = model.rsquared
p_values = model.pvalues


print("Katsayılar:")
print(coefficients)
print("\nR-kare değeri:", r_squared)
print("\nP-değerleri:")
print(p_values)
