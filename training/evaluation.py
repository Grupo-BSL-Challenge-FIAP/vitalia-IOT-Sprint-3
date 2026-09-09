import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def avaliar_modelo():

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "trained", "vitalia_peso_regressor.pkl")
    test_data_path = os.path.join(os.path.dirname(__file__), "dados_teste.csv") # Ajuste conforme seu dataset

    if not os.path.exists(model_path):
        print(f"Modelo não encontrado em: {model_path}. Execute o training.py primeiro.")
        return

    model = joblib.load(model_path)
    print("Modelo carregado com sucesso para avaliação.")

if __name__ == "__main__":
    avaliar_modelo()