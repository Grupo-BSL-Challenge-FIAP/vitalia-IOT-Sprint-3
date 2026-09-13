from api.services.history_service import HistoryService
from api.services.ai_service import predict_pet_status

class AnalysisService:
    def __init__(self):
        self.history_service = HistoryService()

    def analisar_comportamento(self, pet_id: int):
        ultimo = self.history_service.obter_ultimo_registro(pet_id)
        medias = self.history_service.calcular_medias_historicas(pet_id)
        
        if not ultimo or not medias:
            return None
            
        media_historica = medias.get("mediaAtividadePct", 0)
        atividade_atual = ultimo["atividadePct"]
        
        variacao_pct = 0.0
        if media_historica > 0:
            variacao_pct = ((atividade_atual - media_historica) / media_historica) * 100

        if variacao_pct > 5.0:
            tendencia_qualitativa = "CRESCENTE"
        elif variacao_pct < -5.0:
            tendencia_qualitativa = "DECRESCENTE"
        else:
            tendencia_qualitativa = "ESTÁVEL"

        pet_data = {
            "peso_kg": float(ultimo.get("pesoKg", 10.0)),
            "consumo_agua_ml": float(ultimo.get("consumoAguaMl", 500.0)),
            "atividade_diaria_pct": float(atividade_atual),
            "sono_diario_pct": float(ultimo.get("sonoPct", 50))
        }
        
        status_ml, explicabilidade = predict_pet_status(pet_data)

        return {
            "petId": pet_id,
            "mediaHistorica": media_historica,
            "atual": atividade_atual,
            "variacaoPct": round(variacao_pct, 2),
            "tendencia": tendencia_qualitativa,
            "status": status_ml,
            "justificativaNumerica": f"Média histórica de {media_historica}% com registro atual de {atividade_atual}%, resultando em uma variação de {round(variacao_pct, 2)}% ({tendencia_qualitativa})."
        }