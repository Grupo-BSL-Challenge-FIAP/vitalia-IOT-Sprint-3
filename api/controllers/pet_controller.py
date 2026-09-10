from fastapi import APIRouter, HTTPException
from api.schemas.pet_schemas import (
    PetDataInput, QuestionInput, ReportInput, 
    DashboardInput, RecommendationInput, InsightOutput
)
from api.services.report_service import ReportService
from api.services.recommendation_service import RecommendationService
from api.services.llm_service import generate_pet_insight
import pandas as pd
import joblib
import os
from api.security import verificar_autenticacao
    
from api.services.analysis_service import AnalysisService

from api.services.history_service import HistoryService

router = APIRouter(prefix="/api/ai/pets", tags=["Pets AI"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/trained/vitalia_rf_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../../models/trained/scaler.pkl') 

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception:
    model = None
    scaler = None

STATUS_MAP = {0: "NORMAL", 1: "ATENÇÃO", 2: "ALERTA"}

from api.services.history_service import HistoryService

history_service = HistoryService()

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
async def get_pet_trends_get(pet_id: str):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID do pet inválido.")
        
    tendencias = history_service.calcular_tendencias(pid_int)
    medias = history_service.calcular_medias_historicas(pid_int)
    
    if "tendencia" in tendencias:
        raise HTTPException(status_code=404, detail="Dados insuficientes para calcular tendências.")
        
    return {
        "pet_id": pet_id,
        "tendencia_peso": tendencias["tendenciaPeso"],
        "variacao_peso": tendencias["variacaoPeso"],
        "media_historica_peso": medias.get("mediaPesoKg"),
        "historico_recente_peso": [tendencias["registroAnterior"], tendencias["registroAtual"]]
    }

analysis_service = AnalysisService()

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