import os
import joblib
import pandas as pd
from google import genai

MODEL_PATH = "models/trained/vitalia_rf_model.pkl"
SCALER_PATH = "models/trained/scaler.pkl"  # Ajuste o caminho se necessário

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

def generate_gemini_insight(data_dict: dict, status_text: str):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        prompt = f"""
        Você é a Vitalia AI, uma assistente veterinária de inteligência artificial.
        O status predito para o pet é: {status_text}.
        Métricas: {data_dict}
        Forneça um insight curto e acolhedor para o tutor.
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"Insight indisponível no momento: {str(e)}"