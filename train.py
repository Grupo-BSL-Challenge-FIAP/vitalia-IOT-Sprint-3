import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


data_path = "data/processed/pets_dataset_ready.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset não encontrado em {data_path}")

df = pd.read_csv(data_path)


features = ["peso_kg", "atividade_diaria_pct", "sono_diario_pct", "consumo_agua_ml"]
target = "status"

X = df[features]
y = df[target]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)

os.makedirs("models/trained", exist_ok=True)
joblib.dump(pipeline, "models/trained/vitalia_pipeline.pkl")

print("Pipeline treinado e salvo com sucesso em models/trained/vitalia_pipeline.pkl!")