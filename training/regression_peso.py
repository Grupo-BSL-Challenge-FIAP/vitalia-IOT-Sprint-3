import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def treinar_regressao_peso():
    print("--- Iniciando Treinamento de Regressão: Tendência de Peso ---")
    
    data_path = "data/processed/pets_dataset_ready.csv"
    if not os.path.exists(data_path):
        print(f"Erro: Dataset não encontrado em {data_path}. Execute o preprocessing.py primeiro.")
        return
        
    df = pd.read_csv(data_path)
    
    df = df.dropna(subset=['peso_kg'])
    
    features = ['idade_anos', 'atividade_diaria_pct', 'sono_diario_pct', 'consumo_agua_ml']
    X = df[features]
    y = df['peso_kg']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Métricas de Avaliação:")
    print(f"- MAE (Erro Médio Absoluto): {mae:.2f} kg")
    print(f"- RMSE (Raiz do Erro Quadrático Médio): {rmse:.2f} kg")
    print(f"- Coeficiente de Determinação (R²): {r2:.2f}")
    
    os.makedirs("models/trained", exist_ok=True)
    model_path = "models/trained/vitalia_peso_regressor.pkl"
    joblib.dump(model, model_path)
    print(f"[OK] Modelo de regressão de peso salvo com sucesso em: {model_path}")
    
    dados_recentes = pd.DataFrame([[3.0, 75.0, 60.0, 500.0]], columns=features)
    peso_previsto = model.predict(dados_recentes)[0]
    print(f"\n[Exemplo de Previsão] Peso estimado para o próximo ciclo/dias: {peso_previsto:.2f} kg")

if __name__ == "__main__":
    treinar_regressao_peso()