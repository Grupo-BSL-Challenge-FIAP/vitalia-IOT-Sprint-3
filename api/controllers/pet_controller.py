from fastapi import APIRouter, HTTPException
from api.schemas.pet_schemas import (
    PetDataInput, QuestionInput, ReportInput, 
    DashboardInput, RecommendationInput, InsightOutput
)
from api.services.report_service import ReportService
from api.recommendation_service import RecommendationService
from api.services.llm_service import generate_pet_insight
import pandas as pd
import joblib
import os
from api.security import verificar_autenticacao

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

@router.post("/{pet_id}/dashboard")
async def get_pet_dashboard(pet_id: str, payload: DashboardInput):
    try:
        media_ativ = sum(payload.historico_atividades) / len(payload.historico_atividades) if payload.historico_atividades else 0
        media_sono = sum(payload.historico_sonos) / len(payload.historico_sonos) if payload.historico_sonos else 0
        media_agua = sum(payload.historico_aguas) / len(payload.historico_aguas) if payload.historico_aguas else 0
        
        atividade_atual = payload.historico_atividades[-1] if payload.historico_atividades else 0
        
        status_geral = "NORMAL"
        reducao_pct = 0.0

        if atividade_atual < media_ativ and media_ativ > 0:
            reducao_pct = ((media_ativ - atividade_atual) / media_ativ) * 100
            if reducao_pct >= 30.0:
                status_geral = "ATENÇÃO"

        if media_ativ < 20 or media_sono > 90:
            status_geral = "ALERTA"

        return {
            "pet_id": pet_id,
            "status_geral": status_geral,
            "indicadores_consolidados": {
                "media_atividade_pct": round(media_ativ, 2),
                "atividade_atual_pct": round(atividade_atual, 2),
                "reducao_atividade_pct": round(reducao_pct, 2),
                "media_sono_pct": round(media_sono, 2),
                "media_consumo_agua_ml": round(media_agua, 2),
                "total_registros_analisados": len(payload.historico_atividades)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar dados do dashboard: {str(e)}")

@router.post("/{pet_id}/predict", response_model=InsightOutput)
async def predict_pet_status(pet_id: str, data: PetDataInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo de IA não carregado.")
    
    features_df = pd.DataFrame([{
        "idade_anos": data.idade_anos,
        "peso_kg": data.peso_kg,
        "atividade_diaria_pct": data.atividade_diaria_pct,
        "sono_diario_pct": data.sono_diario_pct,
        "consumo_agua_ml": data.consumo_agua_ml
    }])
    
    if scaler:
        features_transformadas = scaler.transform(features_df)
        features_modelo = features_transformadas[:, [2, 1, 3]]
        prediction = model.predict(features_modelo)[0]
    else:
        features_simples = pd.DataFrame([{
            "atividade": data.atividade_diaria_pct,
            "peso_var": data.peso_kg,
            "sono": 1 if data.sono_diario_pct > 50 else 0
        }])
        prediction = model.predict(features_simples)[0]
    status_text = STATUS_MAP.get(prediction, "DESCONHECIDO")
   
    fatores_explicacao = []
    if data.atividade_diaria_pct < 20:
        fatores_explicacao.append(f"Atividade física baixa ({data.atividade_diaria_pct}%)")
    if data.sono_diario_pct > 90:
        fatores_explicacao.append(f"Horas de sono elevadas ({data.sono_diario_pct}%)")
    if data.consumo_agua_ml < 250:
        fatores_explicacao.append(f"Baixo consumo de água ({data.consumo_agua_ml} ml)")
        
    if not fatores_explicacao:
        fatores_explicacao.append("Todos os indicadores comportamentais estão dentro dos parâmetros normais.")

    dados_explicabilidade = {
        "fatores_principais": fatores_explicacao,
        "resumo_analise": f"Avaliação realizada considerando peso de {data.peso_kg}kg, idade de {data.idade_anos} anos e métricas diárias."
    }

    insight_gerado = generate_pet_insight(data.model_dump(), status_text)
    
    return InsightOutput(
        pet_id=pet_id, 
        status=status_text, 
        mensagem_alerta="Classificação baseada em modelo Random Forest.",
        explicabilidade=dados_explicabilidade,
        insight_ia=insight_gerado
    )

@router.post("/{pet_id}/recommendations")
async def get_pet_recommendations(pet_id: str, payload: RecommendationInput):
    try:
        recomendacoes = []
        d = payload.dados_atuais
        
        if d.atividade_diaria_pct < 20:
            recomendacoes.append({"categoria": "Atividade Física", "acao": "Introduzir brincadeiras interativas de curta duração", "justificativa": f"A atividade diária registrou apenas {d.atividade_diaria_pct}%."})
        if d.sono_diario_pct > 90:
            recomendacoes.append({"categoria": "Bem-estar e Descanso", "acao": "Monitorar sinais de apatia contínua", "justificativa": f"O tempo de sono esteve elevado em {d.sono_diario_pct}%."})
        if d.atividade_diaria_pct < 50:
            recomendacoes.append({
                "categoria": "Atividade Física", 
                "acao": "Acompanhar o nível de atividade do pet nos próximos dias.", 
                "justificativa": "Atividade abaixo da média histórica esperada."
            })
        if d.consumo_agua_ml < 250:
            recomendacoes.append({"categoria": "Hidratação", "acao": "Espalhar mais potes de água pela casa", "justificativa": f"Consumo hídrico registrando {d.consumo_agua_ml} ml."})
            
        if not recomendacoes:
            recomendacoes.append({"categoria": "Manutenção da Rotina", "acao": "Manter a rotina atual", "justificativa": "Parâmetros equilibrados."})

        return {"pet_id": pet_id, "total_recomendacoes": len(recomendacoes), "recomendacoes": recomendacoes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar recomendações: {str(e)}")

@router.post("/{pet_id}/ask")
async def ask_pet_question(pet_id: str, payload: QuestionInput):
    try:
        from google import genai
        client = genai.Client()
        
        prompt = f"""
        Você é a Vitalia AI, assistente virtual veterinária.
        Responda à pergunta do tutor com base nos dados atuais:
        Idade: {payload.dados_atuais.idade_anos} anos | Peso: {payload.dados_atuais.peso_kg} kg | Atividade: {payload.dados_atuais.atividade_diaria_pct}%
        Pergunta: "{payload.pergunta}"
        Responda de forma clara em até 3 frases.
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return {"pet_id": pet_id, "pergunta": payload.pergunta, "resposta_ia": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na IA: {str(e)}")

@router.post("/{pet_id}/report")
async def generate_pet_report(pet_id: str, payload: ReportInput):
    try:
        from google import genai
        client = genai.Client()
        prompt = f"Gere um relatório {payload.periodo} para o pet {pet_id} com base nos dados: {payload.dados_atuais.model_dump()}"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return {"pet_id": pet_id, "periodo": payload.periodo, "relatorio_gerado": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pet_id}/relatorio/{tipo}/exportar")
def exportar_relatorio_pet(pet_id: int, tipo: str):
    if tipo not in ["diario", "semanal", "mensal"]:
        raise HTTPException(status_code=400, detail="Tipo inválido.")
    relatorio = ReportService.gerar_relatorio(pet_id, tipo)
    return {"mensagem": "Exportado com sucesso!", "formato": "JSON", "conteudo": relatorio}

@router.get("/{pet_id}/insights")
async def get_pet_insights_get(pet_id: str):
    return {"pet_id": pet_id, "status_analise": "Normal", "insights": "Comportamento dentro da normalidade."}

@router.get("/{pet_id}/analysis")
async def get_pet_analysis_get(pet_id: str):
    resultado = RecommendationService.gerar_recomendacao({"atividade_diaria_pct": 75, "peso_variacao_pct": 0, "sono_status": "estável"})
    return {"pet_id": pet_id, "analise_comportamental": resultado}

@router.get("/{pet_id}/trends")
async def get_pet_trends_get(pet_id: str):
    return {"pet_id": pet_id, "tendencia_atividade": "Estável", "tendencia_sono": "Regular", "historico_analisado": [15.0, 18.0, 16.0, 15.0]}