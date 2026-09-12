import os
import pandas as pd

class HistoryService:
    def __init__(self, data_path="data/processed/pets_dataset_ready.csv"):
        self.data_path = data_path

    def _carregar_dados(self):
        if not os.path.exists(self.data_path):
            return pd.DataFrame()
        df = pd.read_csv(self.data_path)
        if 'pet_id' in df.columns:
            df['pet_id'] = df['pet_id'].astype(int)
        return df

    def obter_ultimo_registro(self, pet_id: int):
        df = self._carregar_dados()
        if df.empty:
            return None
        
        pet_df = df[df['pet_id'] == int(pet_id)]
        if pet_df.empty:
            return None
        
        pet_df = pet_df.sort_values(by='data', ascending=False)
        ultimo = pet_df.iloc[0]
        
        return {
            "petId": int(ultimo['pet_id']),
            "data": str(ultimo['data']),
            "pesoKg": float(ultimo['peso_kg']),
            "atividadePct": float(ultimo['atividade_diaria_pct']),
            "sonoPct": float(ultimo['sono_diario_pct']),
            "consumoAguaMl": float(ultimo['consumo_agua_ml'])
        }

    def calcular_medias_historicas(self, pet_id: int):
        df = self._carregar_dados()
        if df.empty:
            return {}
        
        pet_df = df[df['pet_id'] == int(pet_id)]
        if pet_df.empty:
            return {}
        
        return {
            "mediaPesoKg": float(pet_df['peso_kg'].mean().round(2)),
            "mediaAtividadePct": float(pet_df['atividade_diaria_pct'].mean().round(1)),
            "mediaSonoPct": float(pet_df['sono_diario_pct'].mean().round(1)),
            "mediaConsumoAguaMl": float(pet_df['consumo_agua_ml'].mean().round(1))
        }