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
        
        pet_df = pet_df.sort_values(by='data', ascending=False)
        
        historical_df = pet_df.iloc[1:] if len(pet_df) > 1 else pet_df
        
        return {
            "mediaPesoKg": float(historical_df['peso_kg'].mean().round(2)),
            "mediaAtividadePct": float(historical_df['atividade_diaria_pct'].mean().round(1)),
            "mediaSonoPct": float(historical_df['sono_diario_pct'].mean().round(1)),
            "mediaConsumoAguaMl": float(historical_df['consumo_agua_ml'].mean().round(1))
        }
    def prever_regressao_peso(self, pet_id: int):
        df = self._carregar_dados()
        if df.empty:
            return None
        
        pet_df = df[df['pet_id'] == int(pet_id)].sort_values(by='data')
        if len(pet_df) < 2:
            return None 

        X = range(len(pet_df))
        y = pet_df['peso_kg'].values
        
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        X_arr = np.array(X).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_arr, y)
        y_pred = model.predict(X_arr)
        
        proximo_x = np.array([[len(pet_df)]])
        peso_previsto = float(model.predict(proximo_x)[0])
        
        return {
            "petId": pet_id,
            "historico_analisado": len(pet_df),
            "pesoPrevisto": round(peso_previsto, 2),
            "mae": float(mean_absolute_error(y, y_pred)),
            "mse": float(mean_squared_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": float(r2_score(y, y_pred))
        }