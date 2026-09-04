from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd

from api.services.report_service import ReportService

from api.recommendation_service import RecommendationService

from api.services.llm_service import generate_pet_insight

import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("APP_ENV", "development")
SECRET_KEY = os.getenv("API_SECRET_KEY", "default-secret")

app = FastAPI(title="Vitalia AI API", description="API de Inteligência Artificial para Pets com LLM", version="1.2")

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/trained/vitalia_rf_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None

class QuestionInput(BaseModel):
    pergunta: str
    dados_atuais: PetDataInput 
    
class ReportInput(BaseModel):
    periodo: str  
    dados_atuais: PetDataInput
    historico_recente: list[float] = [15.0, 18.0, 16.0, 15.0] 

    
class PetDataInput(BaseModel):
    idade_anos: float
    peso_kg: float
    atividade_diaria_pct: float
    sono_diario_pct: float
    consumo_agua_ml: float

class InsightOutput(BaseModel):
    pet_id: str
    status: str
    mensagem_alerta: str
    explicabilidade: dict 
    insight_ia: str
    
class RecommendationInput(BaseModel):
    dados_atuais: PetDataInput

STATUS_MAP = {0: "NORMAL", 1: "ATENÇÃO", 2: "ALERTA"}

class DashboardInput(BaseModel):
    historico_atividades: list[float]
    historico_sonos: list[float]
    historico_aguas: list[float]

@app.post("/api/ai/pets/{pet_id}/recommendations")
async def get_pet_recommendations(pet_id: str, payload: RecommendationInput):
    try:
        recomendacoes = []
        
        if payload.dados_atuais.atividade_diaria_pct < 20:
            recomendacoes.append({
                "categoria": "Atividade Física",
                "acao": "Introduzir brincadeiras interativas de curta duração (como buscar a bolinha)",
                "justificativa": f"A atividade diária registrou apenas {payload.dados_atuais.atividade_diaria_pct}%, indicando sedentarismo."
            })
            
        if payload.dados_atuais.sono_diario_pct > 90:
            recomendacoes.append({
                "categoria": "Bem-estar e Descanso",
                "acao": "Monitorar sinais de apatia contínua e verificar se há dor ou desconforto físico",
                "justificativa": f"O tempo de sono esteve elevado em {payload.dados_atuais.sono_diario_pct}% do período."
            })
            
        if payload.dados_atuais.consumo_agua_ml < 250:
            recomendacoes.append({
                "categoria": "Hidratação",
                "acao": "Espalhar mais potes de água fresca pela casa ou adicionar sachê na ração para estimular o consumo",
                "justificativa": f"O consumo hídrico ficou abaixo do ideal, registrando {payload.dados_atuais.consumo_agua_ml} ml."
            })
            
        if not recomendacoes:
            recomendacoes.append({
                "categoria": "Manutenção da Rotina",
                "acao": "Manter a rotina atual de exercícios e alimentação",
                "justificativa": "Todos os parâmetros do pet encontram-se equilibrados e dentro da normalidade."
            })

        return {
            "pet_id": pet_id,
            "total_recomendacoes": len(recomendacoes),
            "recomendacoes": recomendacoes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar recomendações: {str(e)}")


@app.post("/api/ai/pets/{pet_id}/dashboard")
async def get_pet_dashboard(pet_id: str, payload: DashboardInput):
    try:
        media_atividade = sum(payload.historico_atividades) / len(payload.historico_atividades) if payload.historico_atividades else 0
        media_sono = sum(payload.historico_sonos) / len(payload.historico_sonos) if payload.historico_sonos else 0
        media_agua = sum(payload.historico_aguas) / len(payload.historico_aguas) if payload.historico_aguas else 0
        
        status_geral = "ALERTA" if media_atividade < 20 or media_sono > 90 else "NORMAL"
        
        return {
            "pet_id": pet_id,
            "status_geral": status_geral,
            "indicadores_consolidados": {
                "media_atividade_pct": round(media_atividade, 2),
                "media_sono_pct": round(media_sono, 2),
                "media_consumo_agua_ml": round(media_agua, 2),
                "total_registros_analisados": len(payload.historico_atividades)
            },
            "tendencia": "Estável com picos de repouso" if status_geral == "ALERTA" else "Comportamento regular dentro da normalidade"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar dados do dashboard: {str(e)}")

@app.post("/api/ai/pets/{pet_id}/predict", response_model=InsightOutput)
async def predict_pet_status(pet_id: str, data: PetDataInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo de IA não carregado.")
    

    features_df = pd.DataFrame([{
        "atividade": dados.atividade_diaria_pct,
        "peso_var": dados.peso_kg,
        "sono": 1 if dados.sono_diario_pct > 50 else 0
    }])
    prediction = model.predict(features_df)[0]
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
    
@app.post("/api/ai/pets/{pet_id}/ask")
async def ask_pet_question(pet_id: str, payload: QuestionInput):
    try:
        from google import genai
        import os
        
        client = genai.Client()
        
        prompt = f"""
        Você é a Vitalia AI, uma assistente virtual veterinária e de acompanhamento pet amigável.
        Responda à pergunta do tutor com base estritamente nos dados atuais do pet informados abaixo.
        
        Dados do Pet ({pet_id}):
        - Idade: {payload.dados_atuais.idade_anos} anos
        - Peso: {payload.dados_atuais.peso_kg} kg
        - Atividade diária: {payload.dados_atuais.atividade_diaria_pct}%
        - Sono diário: {payload.dados_atuais.sono_diario_pct}%
        - Consumo de água: {payload.dados_atuais.consumo_agua_ml} ml
        
        Pergunta do Tutor: "{payload.pergunta}"
        
        Responda de forma clara, acolhedora e direta, em até 3 frases.
        """
        
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        
        return {
            "pet_id": pet_id,
            "pergunta": payload.pergunta,
            "resposta_ia": response.text.strip()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar pergunta com a IA: {str(e)}")
    
    
@app.post("/api/ai/pets/{pet_id}/report")
async def generate_pet_report(pet_id: str, payload: ReportInput):
    try:
        from google import genai
        
        client = genai.Client()
        
    
        prompt = f"""
        Você é a Vitalia AI, uma assistente veterinária de inteligência artificial.
        Gere um relatório {payload.periodo} formal e acolhedor para o tutor sobre o pet {pet_id}.
        
        Métricas Atuais:
        - Idade: {payload.dados_atuais.idade_anos} anos
        - Peso: {payload.dados_atuais.peso_kg} kg
        - Atividade diária: {payload.dados_atuais.atividade_diaria_pct}%
        - Sono diário: {payload.dados_atuais.sono_diario_pct}%
        - Consumo de água: {payload.dados_atuais.consumo_agua_ml} ml
        
        Estruture o relatório em:
        1. Resumo do Período
        2. Destaques Comportamentais
        3. Recomendações para o Tutor
        
        Mantenha um tom profissional, claro e objetivo.
        """
        
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        
        return {
            "pet_id": pet_id,
            "periodo": payload.periodo,
            "relatorio_gerado": response.text.strip()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar relatório com a IA: {str(e)}")
    
    
@app.get("/pets/{pet_id}/relatorio/{tipo}/exportar")
def exportar_relatorio_pet(pet_id: int, tipo: str):
    if tipo not in ["diario", "semanal", "mensal"]:
        raise HTTPException(status_code=400, detail="Tipo de relatório inválido. Use: diario, semanal ou mensal.")
    
    relatorio = ReportService.gerar_relatorio(pet_id, tipo)
    return {
        "mensagem": "Relatório exportado com sucesso!",
        "formato": "JSON",
        "conteudo": relatorio
    }

@app.post("/pets/recomendacoes")
def obter_recomendacao_pet(dados: dict):
    resultado = RecommendationService.gerar_recomendacao(dados)
    return resultado

from api.services.report_service import ReportService

@app.get("/api/ai/pets/{pet_id}/insights")
async def get_pet_insights_get(pet_id: str):
    return {
        "pet_id": pet_id,
        "status_analise": "Normal",
        "insights": "Comportamento dentro da normalidade para o histórico recente do pet."
    }

@app.get("/api/ai/pets/{pet_id}/analysis")
async def get_pet_analysis_get(pet_id: str):
    resultado_padrao = RecommendationService.gerar_recomendacao({
        "atividade_diaria_pct": 75,
        "peso_variacao_pct": 0,
        "sono_status": "estável"
    })
    return {
        "pet_id": pet_id,
        "analise_comportamental": resultado_padrao
    }

@app.get("/api/ai/pets/{pet_id}/report")
async def get_pet_report_get(pet_id: str):
    relatorio_base = ReportService.gerar_relatorio(pet_id, "semanal")
    return {
        "pet_id": pet_id,
        "periodo": "semanal",
        "relatorio": relatorio_base
    }

@app.get("/api/ai/pets/{pet_id}/trends")
async def get_pet_trends_get(pet_id: str):
    return {
        "pet_id": pet_id,
        "tendencia_atividade": "Estável",
        "tendencia_sono": "Regular",
        "historico_analisado": [15.0, 18.0, 16.0, 15.0]
    }