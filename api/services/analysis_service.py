import joblib
import os
import pandas as pd
from api.services.history_service import HistoryService

PIPELINE_PATH = "models/trained/vitalia_pipeline.pkl"

class AnalysisService:
    def __init__(self):
        self.history_service = HistoryService()
        if os.path.exists(PIPELINE_PATH):
            self.pipeline = joblib.load(PIPELINE_PATH)
        else:
            self.pipeline = None

    def analisar_comportamento(self, pet_id: int):
        ultimo = self.history_service.obter_ultimo_registro(pet_id)
        medias = self.history_service.calcular_medias_historicas(pet_id)
        
        if not ultimo or not medias:
            return None
            
        metricas_map = [
            ("peso", float(ultimo.get("pesoKg", 0)), float(medias.get("mediaPesoKg", 0))),
            ("atividade", float(ultimo.get("atividadePct", 0)), float(medias.get("mediaAtividadePct", 0))),
            ("sono", float(ultimo.get("sonoPct", 0)), float(medias.get("mediaSonoPct", 0))),
            ("consumoAgua", float(ultimo.get("consumoAguaMl", 0)), float(medias.get("mediaConsumoAguaMl", 0)))
        ]

        alteracoes = []
        for nome, atual, media_hist in metricas_map:
            var_pct = 0.0
            if media_hist > 0:
                var_pct = round(((atual - media_hist) / media_hist) * 100, 2)
            alteracoes.append({
                "metrica": nome,
                "mediaHistorica": media_hist,
                "atual": atual,
                "variacaoPct": var_pct
            })

        df_input = pd.DataFrame([{
            "idade_anos": float(ultimo.get("idade_anos", 3.0)),
            "peso_kg": float(ultimo.get("pesoKg", 10.0)),
            "atividade_diaria_pct": float(ultimo.get("atividadePct", 50.0)),
            "sono_diario_pct": float(ultimo.get("sonoPct", 50.0)),
            "consumo_agua_ml": float(ultimo.get("consumoAguaMl", 500.0))
        }])

        status_ml = "NORMAL"
        if self.pipeline:
            pred = self.pipeline.predict(df_input)
            val = str(pred[0])
            mapping = {"0": "NORMAL", "1": "ATENÇÃO", "2": "ALERTA"}
            status_ml = mapping.get(val, val)

        media_atividade = float(medias.get("mediaAtividadePct", 0))
        ativ_atual = float(ultimo.get("atividadePct", 0))
        var_ativ = 0.0
        if media_atividade > 0:
            var_ativ = round(((ativ_atual - media_atividade) / media_atividade) * 100, 2)

        tendencia_qualitativa = "CRESCENTE" if var_ativ > 5.0 else ("DECRESCENTE" if var_ativ < -5.0 else "ESTÁVEL")

        return {
            "petId": pet_id,
            "mediaHistorica": media_atividade,
            "atual": ativ_atual,
            "variacaoPct": var_ativ,
            "tendencia": tendencia_qualitativa,
            "status": status_ml,
            "alteracoes": alteracoes,
            "justificativaNumerica": f"Média histórica de atividade de {media_atividade}% com registro atual de {ativ_atual}%, resultando em uma variação de {var_ativ}% ({tendencia_qualitativa})."
        }