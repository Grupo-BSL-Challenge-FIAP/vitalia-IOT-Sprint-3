import pandas as pd
from datetime import datetime
from api.repositories.pet_repository import PetRepository

class HistoryService:
    def __init__(self, repository: PetRepository = None):
        self.repository = repository or PetRepository()

    def _calcular_idade(self, data_nascimento_str):
        try:
            data_nasc = datetime.strptime(str(data_nascimento_str)[:10], "%Y-%m-%d")
            hoje = datetime.now()
            idade = hoje.year - data_nasc.year - ((hoje.month, hoje.day) < (data_nasc.month, data_nasc.day))
            return max(0, idade)
        except Exception:
            return 3.0

    def obter_ultimo_registro(self, pet_id: int):
        pet_info = self.repository.obter_informacoes_pet(pet_id)
        medicoes = self.repository.obter_historico_medicoes(pet_id)
        
        if not medicoes:
            return None
            
        ultimo = medicoes[0]
        idade = self._calcular_idade(pet_info.get("birth_date")) if pet_info else 3.0

        return {
            "petId": int(pet_id),
            "data": str(ultimo.get("measurement_date")),
            "idadeAnos": float(idade),
            "pesoKg": float(ultimo.get("weight_kg", 10.0)),
            "atividadePct": float(ultimo.get("activity_level", 50.0)),
            "sonoPct": float(ultimo.get("sleep_hours", 50.0)),
            "frequenciaCardiacaBpm": float(ultimo.get("heart_rate_bpm", 100.0)),
            "frequenciaRespiratoria": float(ultimo.get("respiratory_rate", 24.0)),
            "temperaturaC": float(ultimo.get("temperature_c", 38.5)),
            "scoreComportamento": float(ultimo.get("behavior_score", 8.0))
        }

    def calcular_medias_historicas(self, pet_id: int):
        medicoes = self.repository.obter_historico_medicoes(pet_id)
        if not medicoes:
            return {}

        df = pd.DataFrame(medicoes)
        historical_df = df.iloc[1:] if len(df) > 1 else df

        return {
            "mediaPesoKg": float(historical_df['weight_kg'].mean().round(2)) if 'weight_kg' in historical_df else 0.0,
            "mediaAtividadePct": float(historical_df['activity_level'].mean().round(1)) if 'activity_level' in historical_df else 0.0,
            "mediaSonoPct": float(historical_df['sleep_hours'].mean().round(1)) if 'sleep_hours' in historical_df else 0.0,
            "mediaTemperaturaC": float(historical_df['temperature_c'].mean().round(1)) if 'temperature_c' in historical_df else 38.5
        }