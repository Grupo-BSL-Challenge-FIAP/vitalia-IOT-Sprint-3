import pandas as pd
import numpy as np
import os

def generate_synthetic_data(filepath: str, num_records: int = 2000):
    """Gera dados simulados de comportamento e saúde de pets para o MVP."""
    np.random.seed(42)
    
    data = {
        'pet_id': np.random.randint(1000, 1100, num_records),
        'idade_anos': np.random.uniform(0.5, 15, num_records).round(1),
        'peso_kg': (5.0 + np.random.uniform(0.5, 3.0, num_records) * np.random.uniform(0.8, 1.5, num_records)).round(2),
        'atividade_diaria_pct': np.random.normal(70, 20, num_records).clip(0, 100).round(1),
        'sono_diario_pct': np.random.normal(60, 15, num_records).clip(0, 100).round(1),
        'consumo_agua_ml': np.random.normal(600, 200, num_records).clip(100, 2000).round(0),
    }
    
    df = pd.DataFrame(data)
    
    condicoes = [
        (df['atividade_diaria_pct'] < 25) | (df['sono_diario_pct'] > 90) | (df['consumo_agua_ml'] < 250),
        (df['atividade_diaria_pct'] < 45) | (df['sono_diario_pct'] > 80) | (df['consumo_agua_ml'] < 400)
    ]
    escolhas = [2, 1]
    df['status'] = np.select(condicoes, escolhas, default=0)
    
    idx_nulos = np.random.choice(df.index, size=50, replace=False)
    df.loc[idx_nulos, 'peso_kg'] = np.nan
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[OK] Dataset bruto gerado em: {filepath}")
    return df

def preprocess_data(raw_filepath: str, processed_filepath: str):
    """Limpa, trata valores nulos e prepara o dataset final."""
    print("Iniciando pré-processamento...")
    df = pd.read_csv(raw_filepath)
    
    df['peso_kg'] = df['peso_kg'].fillna(df['peso_kg'].median())
    
    df = df.drop_duplicates()
    
    features = ['idade_anos', 'peso_kg', 'atividade_diaria_pct', 'sono_diario_pct', 'consumo_agua_ml']
    X = df[features]
    y = df['status']
    

    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    dataset_final = pd.concat([X, y], axis=1)
    dataset_final.to_csv(processed_filepath, index=False)
    
    print(f"[OK] Dataset limpo e processado salvo em: {processed_filepath}")
    return X, y

if __name__ == "__main__":
    RAW_PATH = "data/raw/pets_raw_data.csv"
    PROCESSED_PATH = "data/processed/pets_dataset_ready.csv"
    
    generate_synthetic_data(RAW_PATH)
    preprocess_data(RAW_PATH, PROCESSED_PATH)