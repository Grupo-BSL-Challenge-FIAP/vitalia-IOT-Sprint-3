import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    data_path = "data/processed/pets_dataset_ready.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Dataset não encontrado em {data_path}. Certifique-se de executar o preprocessing.py.")
        return

    features = ["peso_kg", "atividade_diaria_pct", "sono_diario_pct", "consumo_agua_ml"]
    target = "status"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("--- Treinando Pipeline Oficial (Random Forest) ---")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia do Pipeline: {acc * 100:.2f}%")

    os.makedirs("models/trained", exist_ok=True)
    pipeline_path = "models/trained/vitalia_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline salvo com sucesso em {pipeline_path}!")

if __name__ == "__main__":
    main()