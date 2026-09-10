import os
import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas.pet_schemas import (
    PetDataInput, QuestionInput, ReportInput, 
    DashboardInput, RecommendationInput, InsightOutput
)
from api.services.report_service import ReportService
from api.services.recommendation_service import RecommendationService
from api.services.analysis_service import AnalysisService
from api.services.history_service import HistoryService
from api.services.llm_service import llm_service
from api.security import verificar_autenticacao

router = APIRouter(prefix="/api/ai/pets", tags=["Pets AI"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/trained/vitalia_rf_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../../models/trained/scaler.pkl') 
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), '../../models/trained/vitalia_pipeline.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception:
    model = None
    scaler = None
    
try:
    pipeline = joblib.load(PIPELINE_PATH)
except Exception:
    pipeline = None

STATUS_MAP = {0: "NORMAL", 1: "ATENÇÃO", 2: "ALERTA"}

history_service = HistoryService()
analysis_service = AnalysisService()

@router.get("/{pet_id}/insights")
async def get_pet_insights_get(pet_id: str):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID do pet inválido.")
        
    ultimo = history_service.obter_ultimo_registro(pid_int)
    if not ultimo:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
    
    atividade = ultimo["atividadePct"]
    sono = ultimo["sonoPct"]
    
    if atividade < 30 or sono > 85:
        status_analise = "Alerta"
        insights = "Comportamento fora da normalidade. Baixa atividade ou excesso de sono detectado."
    else:
        status_analise = "Normal"
        insights = "Comportamento dentro da normalidade com base no histórico recente."
        
    return {
        "pet_id": pet_id, 
        "status_analise": status_analise, 
        "insights": insights,
        "ultimo_registro": ultimo
    }

@router.get("/{pet_id}/trends")
async def get_pet_trends(pet_id: str):
    """
    Retorna tendências históricas e projeção de regressão de peso.
    Aviso: A previsão atualmente utiliza dados simulados, pois o dispositivo IoT ainda não está disponível.
    """
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID do pet inválido.")
        
    tendencias = history_service.calcular_tendencias(pid_int)
    medias = history_service.calcular_medias_historicas(pid_int)
    
    regressor_path = os.path.join(os.path.dirname(__file__), '../../models/trained/vitalia_peso_regressor.pkl')
    historico_dias = []
    peso_projetado = []
    
    if os.path.exists(regressor_path):
        regressor = joblib.load(regressor_path)
        dias_futuros = np.array(range(1, 8)).reshape(-1, 1)
        previsoes = regressor.predict(dias_futuros)
        historico_dias = [int(d) for d in dias_futuros.flatten()]
        peso_projetado = [round(float(p), 2) for p in previsoes]

    return {
        "pet_id": pet_id,
        "tendencia_peso": tendencias.get("tendenciaPeso"),
        "variacao_peso": tendencias.get("variacaoPeso"),
        "media_historica_peso": medias.get("mediaPesoKg"),
        "aviso": "A previsão atualmente utiliza dados simulados, pois o dispositivo IoT ainda não está disponível.",
        "historico_dias": historico_dias,
        "peso_projetado_kg": peso_projetado
    }

@router.get("/{pet_id}/dashboard")
async def get_pet_dashboard(pet_id: str):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido.")
        
    resultado = analysis_service.analisar_comportamento(pid_int)
    if not resultado:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    return {
        "pet_id": pet_id,
        "status_geral": resultado["status"],
        "indicadores_consolidados": {
            "media_atividade_pct": resultado["mediaHistorica"],
            "atividade_atual_pct": resultado["atual"],
            "reducao_atividade_pct": abs(resultado["variacaoPct"]),
            "justificativa": resultado["justificativaNumerica"]
        }
    }

@router.get("/{pet_id}/analysis")
async def get_pet_analysis_get(pet_id: str):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido.")

    resultado = analysis_service.analisar_comportamento(pid_int)
    if not resultado:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")

    return {
        "pet_id": pet_id,
        "analise_comportamental": resultado
    }

@router.get("/{pet_id}/recommendations")
async def get_pet_recommendations(pet_id: str):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido.")
        
    resultado = analysis_service.analisar_comportamento(pid_int)
    if not resultado:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    recomendacoes = []
    if resultado["status"] == "ALERTA":
        recomendacoes.append({"categoria": "Bem-estar", "acao": "Monitorar sinais clínicos e buscar avaliação", "justificativa": resultado["justificativaNumerica"]})
    elif resultado["status"] == "ATENÇÃO":
        recomendacoes.append({"categoria": "Atividade Física", "acao": "Ajustar rotina de exercícios do pet", "justificativa": resultado["justificativaNumerica"]})
    else:
        recomendacoes.append({"categoria": "Manutenção", "acao": "Manter rotina atual", "justificativa": resultado["justificativaNumerica"]})
        
    return {
        "pet_id": pet_id,
        "status_analise": resultado["status"],
        "total_recomendacoes": len(recomendacoes),
        "recomendacoes": recomendacoes
    }
    
@router.post("/{pet_id}/predict")
async def prever_comportamento(pet_id: str, dados_pet: PetDataInput):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline ML não carregado.")
        
    df_entrada = pd.DataFrame([{
        "peso_kg": dados_pet.peso_kg,
        "atividade_diaria_pct": dados_pet.atividade_diaria_pct,
        "sono_diario_pct": dados_pet.sono_diario_pct,
        "consumo_agua_ml": dados_pet.consumo_agua_ml
    }])
    
    predicao = pipeline.predict(df_entrada)
    
    return {"pet_id": pet_id, "predicao": int(predicao[0])}

@router.post("/{pet_id}/ask")
async def perguntar_ao_pet(pet_id: str, pergunta: str):
    contexto = {
        "identificacao": f"Pet ID: {pet_id}",
        "dados_atuais": "Peso: 12kg, Atividade: 52%",
        "medias_historicas": "Média de atividade: 69.7%",
        "tendencia": "Estável com leve queda",
        "classificacao_ml": "ATENÇÃO",
        "alteracoes_encontradas": "Queda de 40.3% na atividade comparada ao histórico."
    }
    
    resposta = llm_service.answer(contexto, pergunta)
    return {"pet_id": pet_id, "resposta": resposta}