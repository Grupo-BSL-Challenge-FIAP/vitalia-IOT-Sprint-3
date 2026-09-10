from api.services.history_service import HistoryService

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

        status = "NORMAL"
        if atividade_atual < 20 or ultimo["sonoPct"] > 90:
            status = "ALERTA"
        elif variacao_pct <= -30.0:
            status = "ATENÇÃO"

        return {
            "petId": pet_id,
            "mediaHistorica": media_historica,
            "atual": atividade_atual,
            "variacaoPct": round(variacao_pct, 2),
            "status": status,
            "justificativaNumerica": f"Média histórica de {media_historica}% com registro atual de {atividade_atual}%, resultando em uma variação de {round(variacao_pct, 2)}%."
        }