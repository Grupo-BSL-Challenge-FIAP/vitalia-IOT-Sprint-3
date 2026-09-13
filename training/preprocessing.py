import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(filepath: str, num_records: int = 2000):
    """Gera dados simulados de comportamento e saúde de pets com histórico temporal para o MVP."""
    np.random.seed(42)
    
    pet_ids = np.random.randint(1001, 1100, num_records)
    pet_ids[0] = 1000
    
    datas_base = []
    pet_counters = {}
    
    hoje = datetime.now()
    for pid in pet_ids:
        if pid not in pet_counters:
            pet_counters[pid] = hoje - timedelta(days=np.random.randint(10, 30))
        else:
            pet_counters[pid] += timedelta(days=np.random.randint(1, 3))
            if pet_counters[pid] > hoje:
                pet_counters[pid] = hoje - timedelta(days=np.random.randint(0, 5))
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
    """Limpa, trata valores nulos usando mediana por pet (com fallback global) e remove duplicatas de pet_id + data."""
    print("Iniciando pré-processamento com histórico temporal...")
    df = pd.read_csv(raw_filepath)
    
    global_median = df['peso_kg'].median()
    df['peso_kg'] = df.groupby('pet_id')['peso_kg'].transform(lambda x: x.fillna(x.median()))
    df['peso_kg'] = df['peso_kg'].fillna(global_median)
    
    df = df.drop_duplicates(subset=['pet_id', 'data'], keep='last')
    
    if 1000 not in df['pet_id'].values:
        linhas_pet_1000 = pd.DataFrame({
            'pet_id': [1000, 1000, 1000],
            'data': ['2026-08-01', '2026-08-10', '2026-08-20'],
            'idade_anos': [3.0, 3.0, 3.0],
            'peso_kg': [12.5, 12.6, 12.4],
            'atividade_diaria_pct': [75.0, 80.0, 70.0],
            'sono_diario_pct': [50.0, 45.0, 55.0],
            'consumo_agua_ml': [500.0, 550.0, 480.0],
            'status': [0, 0, 0]
        })
        df = pd.concat([df, linhas_pet_1000], ignore_index=True)

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