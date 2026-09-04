from api.services.model_service import ModelService

def testar_pipeline_dados():
    dados_brutos = {
        "atividade_diaria_pct": 80,
        "peso_variacao_pct": 1.5,
        "sono_status": "estável"
    }
    
    print("Iniciando teste do pipeline de preparação e treino...")
    resultado = ModelService.treinar_e_prever(dados_brutos)
    print(f"Dataset processado e modelo treinado com sucesso!")
    print(f"Resultado da Predição: {resultado}")

if __name__ == "__main__":
    testar_pipeline_dados()