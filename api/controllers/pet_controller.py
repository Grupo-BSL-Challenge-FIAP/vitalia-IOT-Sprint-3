from fastapi import APIRouter, HTTPException
from api.schemas.pet_schemas import PetDataInput, AskInput, ReportInput, DashboardInput, RecommendationInput
from api.services.ai_service import predict_pet_status, generate_gemini_insight
from google import genai
import os
from api.services.report_service import ReportService

router = APIRouter(prefix="/api/ai/pets", tags=["Pets AI"])

@router.post("/{pet_id}/predict")
def predict_health(pet_id: str, payload: PetDataInput):
    data_dict = payload.model_dump()
    status_text, explicabilidade = predict_pet_status(data_dict)
    insight = generate_gemini_insight(data_dict, status_text)
    return {
        "pet_id": pet_id,
        "status": status_text,
        "explicabilidade": explicabilidade,
        "insight_ia": insight
    }

@router.post("/{pet_id}/ask")
def ask_pet_question(pet_id: str, payload: AskInput):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Responda à dúvida do tutor sobre o pet {pet_id}: {payload.pergunta}"
        )
        return {"pet_id": pet_id, "resposta": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{pet_id}/report")
def generate_report(pet_id: str, payload: ReportInput):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        prompt = f"Gere um relatório {payload.periodo} para o pet {pet_id} com base nos dados: {payload.dados_atuais.model_dump()}"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return {"pet_id": pet_id, "periodo": payload.periodo, "relatorio_gerado": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{pet_id}/dashboard")
def get_dashboard(pet_id: str, payload: DashboardInput):
    media_ativ = sum(payload.historico_atividades) / len(payload.historico_atividades) if payload.historico_atividades else 0
    media_sono = sum(payload.historico_sonos) / len(payload.historico_sonos) if payload.historico_sonos else 0
    media_agua = sum(payload.historico_aguas) / len(payload.historico_aguas) if payload.historico_aguas else 0
    
    status_geral = "ALERTA" if media_ativ < 20 or media_sono > 90 else "NORMAL"
    return {
        "pet_id": pet_id,
        "status_geral": status_geral,
        "indicadores_consolidados": {
            "media_atividade_pct": round(media_ativ, 2),
            "media_sono_pct": round(media_sono, 2),
            "media_consumo_agua_ml": round(media_agua, 2),
            "total_registros_analisados": len(payload.historico_atividades)
        }
    }

@router.post("/{pet_id}/recommendations")
def get_recommendations(pet_id: str, payload: RecommendationInput):
    recs = []
    d = payload.dados_atuais
    if d.atividade_diaria_pct < 20:
        recs.append({"categoria": "Atividade Física", "acao": "Brincadeiras curtas", "justificativa": f"Atividade baixa ({d.atividade_diaria_pct}%)"})
    if d.sono_diario_pct > 90:
        recs.append({"categoria": "Descanso", "acao": "Monitorar apatia", "justificativa": f"Sono elevado ({d.sono_diario_pct}%)"})
    if not recs:
        recs.append({"categoria": "Rotina", "acao": "Manter rotina", "justificativa": "Parâmetros equilibrados"})
    return {"pet_id": pet_id, "total_recomendacoes": len(recs), "recomendacoes": recs}



router = APIRouter()

@router.get("/pets/{pet_id}/relatorio/{tipo}")
def obter_relatorio_pet(pet_id: int, tipo: str):
    if tipo not in ["diario", "semanal", "mensal"]:
        raise HTTPException(status_code=400, detail="Tipo de relatório inválido. Use: diario, semanal ou mensal.")
    
    relatorio = ReportService.gerar_relatorio(pet_id, tipo)
    return relatorio