import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(filepath: str, num_records: int = 2000):
    """Gera dados simulados de comportamento e saúde de pets com histórico temporal para o MVP."""
    np.random.seed(42)
    
    pet_ids = np.random.randint(1000, 1100, num_records)
    
    datas_base = []
    pet_counters = {}
    
    for pid in pet_ids:
        if pid not in pet_counters:
            pet_counters[pid] = datetime.now() - timedelta(days=np.random.randint(10, 30))
        else:
            pet_counters[pid] += timedelta(days=np.random.randint(1, 3))
        datas_base.append(pet_counters[pid].strftime("%Y-%m-%d"))

    data = {
        'pet_id': pet_ids,
        'data': datas_base,
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
    print(f"[OK] Dataset bruto com histórico gerado em: {filepath}")
    return df

def preprocess_data(raw_filepath: str, processed_filepath: str):
    """Limpa, trata valores nulos, mantém o pet_id e a data para formar o histórico temporal."""
    print("Iniciando pré-processamento com histórico temporal...")
    df = pd.read_csv(raw_filepath)
    
    df['peso_kg'] = df['peso_kg'].fillna(df['peso_kg'].median())
    df = df.drop_duplicates()
    
    df['data'] = pd.to_datetime(df['data'])
    df = df.sort_values(by=['pet_id', 'data'])
    df['data'] = df['data'].dt.strftime('%Y-%m-%d')
    
    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    df.to_csv(processed_filepath, index=False)
    
    print(f"[OK] Dataset processado com histórico salvo em: {processed_filepath}")
    return df

if __name__ == "__main__":
    RAW_PATH = "data/raw/pets_raw_data.csv"
    PROCESSED_PATH = "data/processed/pets_dataset_ready.csv"
    
    generate_synthetic_data(RAW_PATH)
    preprocess_data(RAW_PATH, PROCESSED_PATH)