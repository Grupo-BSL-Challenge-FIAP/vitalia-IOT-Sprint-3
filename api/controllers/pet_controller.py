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
        # Tenta converter o pet_id para inteiro, se aplicável ao seu dataset
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

@router.get("/{pet_id}/analysis")
async def get_pet_analysis_get(pet_id: str):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID do pet inválido.")
        
    ultimo = history_service.obter_ultimo_registro(pid_int)
    medias = history_service.calcular_medias_historicas(pid_int)
    
    if not ultimo:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    return {
        "pet_id": pet_id,
        "ultimo_registro": ultimo,
        "medias_historicas": medias
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
    
history_service = HistoryService()

@router.get("/{pet_id}/dashboard")
async def get_pet_dashboard(pet_id: str):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID do pet inválido.")
        
    ultimo = history_service.obter_ultimo_registro(pid_int)
    medias = history_service.calcular_medias_historicas(pid_int)
    
    if not ultimo:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    atividade_atual = ultimo["atividadePct"]
    media_ativ = medias.get("mediaAtividadePct", 0)
    
    status_geral = "NORMAL"
    reducao_pct = 0.0

    if atividade_atual < media_ativ and media_ativ > 0:
        reducao_pct = ((media_ativ - atividade_atual) / media_ativ) * 100
        if reducao_pct >= 30.0:
            status_geral = "ATENÇÃO"

    if atividade_atual < 20 or ultimo["sonoPct"] > 90:
        status_geral = "ALERTA"

    return {
        "pet_id": pet_id,
        "status_geral": status_geral,
        "indicadores_consolidados": {
            "media_atividade_pct": media_ativ,
            "atividade_atual_pct": atividade_atual,
            "reducao_atividade_pct": round(reducao_pct, 2),
            "media_sono_pct": medias.get("mediaSonoPct", 0),
            "media_consumo_agua_ml": medias.get("mediaConsumoAguaMl", 0),
            "ultimo_registro": ultimo
        }
    }

@router.get("/{pet_id}/recommendations")
async def get_pet_recommendations(pet_id: str):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID do pet inválido.")
        
    ultimo = history_service.obter_ultimo_registro(pid_int)
    
    if not ultimo:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    recomendacoes = []
    d = ultimo
    
    if d["atividadePct"] < 20:
        recomendacoes.append({"categoria": "Atividade Física", "acao": "Introduzir brincadeiras interativas de curta duração", "justificativa": f"A atividade diária registrou apenas {d['atividadePct']}%."})
    if d["sonoPct"] > 90:
        recomendacoes.append({"categoria": "Bem-estar e Descanso", "acao": "Monitorar sinais de apatia contínua", "justificativa": f"O tempo de sono esteve elevado em {d['sonoPct']}%."})
    if d["consumoAguaMl"] < 250:
        recomendacoes.append({"categoria": "Hidratação", "acao": "Espalhar mais potes de água pela casa", "justificativa": f"Consumo hídrico registrando {d['consumoAguaMl']} ml."})
        
    if not recomendacoes:
        recomendacoes.append({"categoria": "Manutenção da Rotina", "acao": "Manter a rotina atual", "justificativa": "Parâmetros equilibrados com base no histórico."})

    return {
        "pet_id": pet_id, 
        "total_recomendacoes": len(recomendacoes), 
        "recomendacoes": recomendacoes
    }