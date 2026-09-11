import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(filepath: str, num_records: int = 2000):
    """Gera dados simulados garantindo perfil fixo por pet (idade e peso base coerentes) com histórico temporal."""
    np.random.seed(42)
    
    unique_pet_ids = np.random.randint(1000, 1100, size=50)
    
    pet_profiles = {}
    for pid in unique_pet_ids:
        pet_profiles[pid] = {
            "idade_anos": round(np.random.uniform(0.5, 12.0), 1),
            "peso_base_kg": round(np.random.uniform(4.0, 25.0), 2)
        }
    
    registros = []
    pet_counters = {pid: datetime.now() - timedelta(days=60) for pid in unique_pet_ids}
    
    for _ in range(num_records):
        pid = np.random.choice(unique_pet_ids)

        pet_counters[pid] += timedelta(days=np.random.randint(1, 3))
        if pet_counters[pid] > datetime.now():
            pet_counters[pid] = datetime.now() - timedelta(days=np.random.randint(0, 5))
        data_str = pet_counters[pid].strftime("%Y-%m-%d")
        
        perfil = pet_profiles[pid]
        
        var_peso = np.random.normal(1.0, 0.03)
        peso_atual = round(perfil["peso_base_kg"] * var_peso, 2)

        atividade = round(float(np.random.normal(70, 20)), 1)
        atividade = np.clip(atividade, 0, 100)
        
        sono = round(float(np.random.normal(60, 15)), 1)
        sono = np.clip(sono, 0, 100)
        
        agua = round(float(np.random.normal(600, 200)), 0)
        agua = np.clip(agua, 100, 2000)
        
        registros.append({
            'pet_id': pid,
            'data': data_str,
            'idade_anos': perfil["idade_anos"],
            'peso_kg': peso_atual,
            'atividade_diaria_pct': atividade,
            'sono_diario_pct': sono,
            'consumo_agua_ml': agua
        })
    
    df = pd.DataFrame(registros)
    
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
    print(f"[OK] Dataset com perfis fixos de pets gerado em: {filepath}")
    return df

def preprocess_data(raw_filepath: str, processed_filepath: str):
    """Limpa, trata valores nulos e organiza o histórico temporal por pet[cite: 5]."""
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