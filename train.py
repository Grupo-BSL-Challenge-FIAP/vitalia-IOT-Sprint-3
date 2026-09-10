import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

print(f"Classes únicas no dataset: {np.unique(y)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação (Multiclass):\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão (3x3):\n", cm)

os.makedirs("models/trained", exist_ok=True)
joblib.dump(pipeline, "models/trained/vitalia_pipeline.pkl")
print("\nPipeline com 3 classes salvo com sucesso em models/trained/vitalia_pipeline.pkl!")