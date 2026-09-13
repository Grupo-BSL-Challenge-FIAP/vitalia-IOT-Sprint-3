import os
import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from api.schemas.pet_schemas import (
    PetDataInput, QuestionInput, 
    RecommendationInput, InsightOutput
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
        "idade_anos": dados_pet.idade_anos,
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
    
    if not resultado_analise:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")

    contexto = {
        "petId": pet_id,
        "dados_atuais": {
            "peso": ultimo.get('pesoKg'),
            "atividade": ultimo.get('atividadePct'),
            "sono": ultimo.get('sonoPct'),
            "consumo_de_agua": ultimo.get('consumoAguaMl')
        },
        "medias_historicas": {
            "peso": medias.get('mediaPesoKg'),
            "atividade": medias.get('mediaAtividadePct'),
            "sono": medias.get('mediaSonoPct'),
            "consumo_de_agua": medias.get('mediaConsumoAguaMl')
        },
        "tendencia": resultado_analise.get("tendencia"),
        "classificacao": resultado_analise.get("status"),
        "alteracoes_identificadas": resultado_analise.get("alteracoes", [])
    }
    
    resposta = llm_service.answer(contexto, pergunta)
    return {"pet_id": pet_id, "resposta": resposta}

@router.get("/{pet_id}/trends")
async def get_pet_trends(pet_id: str, credentials: str = Depends(verificar_autenticacao)):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido.")
        
    resultado = analysis_service.analisar_comportamento(pid_int)
    if not resultado:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    regressao_peso = history_service.prever_regressao_peso(pid_int)
    if not regressao_peso:
        raise HTTPException(status_code=422, detail="Histórico insuficiente para realizar a previsão.")

    return {
        "pet_id": pet_id,
        "tendencia": resultado["tendencia"],
        "variacao_pct": resultado["variacaoPct"],
        "media_historica": resultado["mediaHistorica"],
        "atividade_atual": resultado["atual"],
        "regressao_peso": regressao_peso,
        "documentacao": "A previsão de peso utiliza dados simulados, pois os dados reais do dispositivo IoT ainda não estão disponíveis.",
        "detalhes": resultado["justificativaNumerica"]
    }