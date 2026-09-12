import os
import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from api.schemas.pet_schemas import (
    PetDataInput, QuestionInput, ReportInput, 
    DashboardInput, RecommendationInput, InsightOutput
)
from api.services.analysis_service import AnalysisService
from api.services.history_service import HistoryService
from api.services.llm_service import llm_service
from api.security import verificar_autenticacao

router = APIRouter(prefix="/api/ai/pets", tags=["Pets AI"])

PIPELINE_PATH = os.path.join(os.path.dirname(__file__), '../../models/trained/vitalia_pipeline.pkl')

try:
    pipeline = joblib.load(PIPELINE_PATH)
except Exception:
    pipeline = None
     
STATUS_MAP = {0: "NORMAL", 1: "ATENÇÃO", 2: "ALERTA"}

history_service = HistoryService()
analysis_service = AnalysisService()

@router.get("/{pet_id}/insights")
async def get_pet_insights_get(pet_id: str, credentials: str = Depends(verificar_autenticacao)):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID do pet inválido.")
        
    resultado = analysis_service.analisar_comportamento(pid_int)
    if not resultado:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
    
    status_analise = resultado["status"]
    insights = resultado.get("justificativaNumerica", "Comportamento analisado com base no histórico recente.")
    ultimo = history_service.obter_ultimo_registro(pid_int)
        
    return {
        "pet_id": pet_id, 
        "status_analise": status_analise, 
        "insights": insights,
        "ultimo_registro": ultimo
    }

@router.get("/{pet_id}/dashboard")
async def get_pet_dashboard(pet_id: str, credentials: str = Depends(verificar_autenticacao)):
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

@router.get("/api/ai/pets/{pet_id}/dashboard")
async def dashboard(pet_id: int, credentials = Depends(verificar_autenticacao)):
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
async def get_pet_recommendations(pet_id: str, credentials: str = Depends(verificar_autenticacao)):
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
async def prever_comportamento(pet_id: str, dados_pet: PetDataInput, credentials: str = Depends(verificar_autenticacao)):
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
async def perguntar_ao_pet(pet_id: str, pergunta: str, credentials: str = Depends(verificar_autenticacao)):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido.")
        
    ultimo = history_service.obter_ultimo_registro(pid_int)
    if not ultimo:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    medias = history_service.calcular_medias_historicas(pid_int)
    resultado_analise = analysis_service.analisar_comportamento(pid_int)
    
    contexto = {
        "identificacao": f"Pet ID: {pet_id}",
        "dados_atuais": f"Peso: {ultimo.get('pesoKg')}kg, Atividade: {ultimo.get('atividadePct')}%, Sono: {ultimo.get('sonoPct')}%",
        "medias_historicas": f"Média de atividade: {medias.get('mediaAtividadePct')}%, Média de peso: {medias.get('mediaPesoKg')}kg",
        "classificacao_ml": resultado_analise.get("status"),
        "alteracoes_encontradas": resultado_analise.get("justificativaNumerica")
    }
    
    resposta = llm_service.answer(contexto, pergunta)
    return {"pet_id": pet_id, "resposta": resposta}