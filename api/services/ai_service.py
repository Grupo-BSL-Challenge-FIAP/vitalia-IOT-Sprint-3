import os
import joblib
import pandas as pd

MODEL_PATH = "models/trained/vitalia_rf_model.pkl"
SCALER_PATH = "models/trained/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

def predict_pet_status(data_dict: dict):
    if not model:
        return "NORMAL", {"erro": "Modelo não carregado"}
    
    features_df = pd.DataFrame([data_dict])
    predicao = model.predict(features_df)[0]
    
    explicabilidade = {}
    if data_dict["atividade_diaria_pct"] < 20:
        explicabilidade["atividade_diaria"] = "Baixa atividade detectada"
    if data_dict["sono_diario_pct"] > 90:
        explicabilidade["sono_diario"] = "Tempo de sono elevado"
        
    return str(predicao), explicabilidade