import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class ModelService:
    @staticmethod
    def treinar_e_prever(dados: dict):
        df = pd.DataFrame([{
            "atividade": dados.get("atividade_diaria_pct", 50),
            "peso_var": dados.get("peso_variacao_pct", 0),
            "sono": 1 if dados.get("sono_status") == "estável" else 0,
            "alerta": 0
        }])
        
        features = ["atividade", "peso_var", "sono"]
        X = df[features]
        y = df["alerta"]
        
        modelo = RandomForestClassifier(n_estimators=10, random_state=42)
        modelo.fit(X, y)
        
        predicao = modelo.predict(X)[0]
        return "Estável" if predicao == 0 else "Requer Atenção"