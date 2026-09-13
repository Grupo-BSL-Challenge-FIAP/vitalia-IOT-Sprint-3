import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_and_save_pipeline(processed_filepath: str, model_output_path: str):
    """Treina o pipeline unificado com StandardScaler e RandomForest, usando stratify."""
    print("Iniciando treinamento do modelo e criação do pipeline...")
    df = pd.read_csv(processed_filepath)
    
    features = ["idade_anos", "peso_kg", "atividade_diaria_pct", "sono_diario_pct", "consumo_agua_ml"]
    X = df[features]
    y = df['status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"[OK] Pipeline treinado e salvo com sucesso em: {model_output_path}")

if __name__ == "__main__":
    PROCESSED_PATH = "data/processed/pets_dataset_ready.csv"
    MODEL_PATH = "models/trained/vitalia_pipeline.pkl"
    
    train_and_save_pipeline(PROCESSED_PATH, MODEL_PATH)