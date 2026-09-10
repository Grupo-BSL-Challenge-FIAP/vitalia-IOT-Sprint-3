import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

np.random.seed(42)
dias = np.arange(1, 31) 

pesos_base = 12.0 + np.sin(dias / 5.0) * 0.5 + np.random.normal(0, 0.1, len(dias))

X = dias.reshape(-1, 1)
y = pesos_base

model = LinearRegression()
model.fit(X, y)

os.makedirs("models/trained", exist_ok=True)
joblib.dump(model, "models/trained/vitalia_peso_regressor.pkl")
print("Regressor de peso com histórico temporal salvo com sucesso!")