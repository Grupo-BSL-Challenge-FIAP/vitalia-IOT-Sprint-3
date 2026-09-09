class RecommendationService:
    @staticmethod
    def gerar_recomendacao(dados: dict):
        atividade = dados.get("atividade_diaria_pct", 75)
        peso_variacao = dados.get("peso_variacao_pct", 0)
        sono_status = dados.get("sono_status", "estável")
        
        if atividade < 70 and peso_variacao > 0:
            analise = f"Atividade ↓ | Peso ↑ | Sono → {sono_status}"
            recomendacao = (
                "Considere aumentar gradualmente a rotina de atividades do pet, "
                "caso isso esteja de acordo com a orientação veterinária."
            )
            motivo = "A recomendação foi gerada devido à queda de atividade física associada ao ganho de peso no histórico recente."
        else:
            analise = f"Atividade: {atividade}% | Peso: {peso_variacao}% | Sono: {sono_status}"
            recomendacao = "Os indicadores do pet estão equilibrados. Mantenha a rotina atual."
            motivo = "Nenhuma alteração crítica foi identificada nos dados recentes."

        return {
            "dados_analisados": analise,
            "recomendacao": recomendacao,
            "motivo_explicacao": motivo,
            "aviso_medico": "Aviso: Esta ferramenta tem caráter informativo e não substitui a consulta veterinária profissional."
        }