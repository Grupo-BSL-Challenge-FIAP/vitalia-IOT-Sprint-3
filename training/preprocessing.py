import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(filepath: str, num_records: int = 2000):
    """Gera dados simulados de comportamento e saúde de pets com histórico temporal para o MVP."""
    np.random.seed(42)
    
    pet_ids = np.random.randint(1001, 1100, num_records)
    
    for i in range(10):
        pet_ids[i] = 1000
    
    unique_pets = np.unique(pet_ids)
    pet_ages = {pid: round(float(np.random.uniform(0.5, 15)), 1) for pid in unique_pets}
    pet_ages[1000] = 3.0
    idades_fixas = [pet_ages[pid] for pid in pet_ids]
    
    pet_base_weights = {pid: round(float(np.random.uniform(4.0, 25.0)), 2) for pid in unique_pets}
    pet_base_weights[1000] = 12.5  
    pesos_variados = [
        round(float(np.clip(pet_base_weights[pid] + np.random.normal(0, 0.2), 1.0, 50.0)), 2) 
        for pid in pet_ids
    ]
    
    datas_base = []
    pet_counters = {}
    
    hoje = datetime.now()
    for i, pid in enumerate(pet_ids):
        if pid == 1000:
            data_pet_1000 = hoje - timedelta(days=(15 - i))
            datas_base.append(data_pet_1000.strftime("%Y-%m-%d"))
        else:
            if pid not in pet_counters:
                pet_counters[pid] = hoje - timedelta(days=np.random.randint(10, 30))
            else:
                pet_counters[pid] += timedelta(days=np.random.randint(1, 3))
                if pet_counters[pid] > hoje:
                    pet_counters[pid] = hoje - timedelta(days=np.random.randint(0, 5))
            datas_base.append(pet_counters[pid].strftime("%Y-%m-%d"))

    atividade_diaria = [round(float(np.clip(np.random.normal(70, 10), 10, 100)), 2) for _ in range(num_records)]
    sono_diario = [round(float(np.clip(np.random.normal(50, 10), 10, 90)), 2) for _ in range(num_records)]
    consumo_agua = [round(float(np.clip(np.random.normal(500, 100), 100, 2000)), 2) for _ in range(num_records)]
    status = np.random.choice([0, 1, 2], size=num_records, p=[0.7, 0.2, 0.1])
    
    df = pd.DataFrame({
        'pet_id': pet_ids,
        'data': datas_base,
        'idade_anos': idades_fixas,
        'peso_kg': pesos_variados,
        'atividade_diaria_pct': atividade_diaria,
        'sono_diario_pct': sono_diario,
        'consumo_agua_ml': consumo_agua,
        'status': status
    })
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def preprocess_data(raw_filepath: str, processed_filepath: str):
    """Limpa, trata valores nulos usando mediana por pet (com fallback global) e remove duplicatas de pet_id + data."""
    print("Iniciando pré-processamento com histórico temporal...")
    df = pd.read_csv(raw_filepath)
    
    global_median = df['peso_kg'].median()
    df['peso_kg'] = df.groupby('pet_id')['peso_kg'].transform(lambda x: x.fillna(x.median()))
    df['peso_kg'] = df['peso_kg'].fillna(global_median)
    
    df = df.drop_duplicates(subset=['pet_id', 'data'], keep='last')
    
    mask_1000 = df['pet_id'] == 1000
    if not mask_1000.any() or mask_1000.sum() < 5:
        df = df[df['pet_id'] != 1000]
        linhas_pet_1000 = pd.DataFrame({
            'pet_id': [1000, 1000, 1000, 1000, 1000],
            'data': ['2026-08-01', '2026-08-05', '2026-08-10', '2026-08-15', '2026-08-20'],
            'idade_anos': [3.0, 3.0, 3.0, 3.0, 3.0],
            'peso_kg': [12.5, 12.6, 12.4, 12.5, 12.7],
            'atividade_diaria_pct': [75.0, 80.0, 70.0, 78.0, 82.0],
            'sono_diario_pct': [50.0, 45.0, 55.0, 48.0, 52.0],
            'consumo_agua_ml': [500.0, 550.0, 480.0, 520.0, 530.0],
            'status': [0, 0, 0, 0, 0]
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