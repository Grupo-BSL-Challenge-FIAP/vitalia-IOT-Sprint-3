import os
import joblib
import pandas as pd

MODEL_PATH = "models/trained/vitalia_pipeline.pkl"

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception:
    pipeline = None

def predict_pet_status(data_dict: dict):
    if not pipeline:
        return "NORMAL", {"erro": "Pipeline não carregado"}
    
    features_df = pd.DataFrame([data_dict])
    predicao = pipeline.predict(features_df)[0]
    
    explicabilidade = {}
    if data_dict.get("atividade_diaria_pct", 50) < 20:
        explicabilidade["atividade_diaria"] = "Baixa atividade detectada"
    if data_dict.get("sono_diario_pct", 50) > 90:
        explicabilidade["sono_diario"] = "Tempo de sono elevado"
        
    return str(predicao), explicabilidade