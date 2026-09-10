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

import os
import joblib
import pandas as pd

router = APIRouter(prefix="/api/ai/pets", tags=["Pets AI"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/trained/vitalia_rf_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../../models/trained/scaler.pkl') 

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception:
    model = None
    scaler = None
    
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), '../../models/trained/vitalia_pipeline.pkl')

try:
    pipeline = joblib.load(PIPELINE_PATH)
except Exception:
    pipeline = None

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


@router.get("/{pet_id}/trends")
async def obter_tendencia_peso(pet_id: str):
    """
    Retorna a tendência de regressão de peso do pet baseada em histórico.
    Aviso: A previsão atualmente utiliza dados simulados, pois o dispositivo IoT ainda não está disponível.
    """
    regressor_path = "models/trained/vitalia_peso_regressor.pkl"
    if not os.path.exists(regressor_path):
        raise HTTPException(status_code=500, detail="Regressor não encontrado.")
    
    model = joblib.load(regressor_path)
    
    dias_futuros = np.array(range(1, 8)).reshape(-1, 1)
    previsoes = model.predict(dias_futuros)
    
    return {
        "pet_id": pet_id,
        "aviso": "A previsão atualmente utiliza dados simulados, pois o dispositivo IoT ainda não está disponível.",
        "historico_dias": [int(d) for d in dias_futuros.flatten()],
        "peso_projetado_kg": [round(float(p), 2) for p in previsoes]
    }
    
def gerar_recomendacao_com_historico(pet_id: str, valor_atual: float, historico_valores: list):
    """
    Calcula a média histórica do pet, compara com o valor atual,
    determina a variação e gera uma recomendação com justificativa numérica.
    """
    if not historico_valores:
        return {
            "historico_medio": None,
            "atual": valor_atual,
            "variacao_pct": None,
            "recomendacao": "Dados históricos insuficientes para comparação."
        }
    
    media_historica = sum(historico_valores) / len(historico_valores)
    variacao_pct = ((valor_atual - media_historica) / media_historica) * 100
    
    # Lógica de recomendação baseada na variação
    if variacao_pct < -20:
        recomendacao = f"Atividade abaixo da média histórica esperada. Histórico: {round(media_historica, 1)}%, Atual: {round(valor_atual, 1)}%, Variação: {round(variacao_pct, 1)}%. Acompanhar o nível de atividade do pet nos próximos dias."
    elif variacao_pct > 20:
        recomendacao = f"Atividade acima da média histórica. Histórico: {round(media_historica, 1)}%, Atual: {round(valor_atual, 1)}%, Variação: {round(variacao_pct, 1)}%."
    else:
        recomendacao = f"Métricas dentro da normalidade comparadas ao histórico de {round(media_historica, 1)}%."
        
    return {
        "historico_medio": round(media_historica, 2),
        "atual": valor_atual,
        "variacao_pct": round(variacao_pct, 2),
        "recomendacao": recomendacao
    }