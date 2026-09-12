import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

os.environ["API_SECRET_KEY"] = "token-compartilhado-123"
os.environ["GEMINI_API_KEY"] = "fake-key-para-testes"

from api.main import app

client = TestClient(app)
HEADERS = {"Authorization": "Bearer token-compartilhado-123"}

VALID_PAYLOAD = {
    "idade_anos": 3.5,
    "peso_kg": 15.0,
    "atividade_diaria_pct": 15.0,
    "sono_diario_pct": 95.0,
    "consumo_agua_ml": 200.0
}
QUERY_PARAMS = "?idade_anos=3.5&peso_kg=15.0&atividade_diaria_pct=15.0&sono_diario_pct=95.0&consumo_agua_ml=200.0"

def criar_mock_response(status, categoria, acao, justificativa):
    mock_response = MagicMock()
    json_text = f'{{"pet_id": "1000", "status_analise": "{status}", "total_recomendacoes": 1, "recomendacoes": [{{"categoria": "{categoria}", "acao": "{acao}", "justificativa": "{justificativa}"}}]}}'
    mock_response.text = json_text
    
    mock_parsed = MagicMock()
    mock_parsed.status_analise = status
    mock_parsed.pet_id = "1000"
    mock_parsed.total_recomendacoes = 1
    
    rec = MagicMock()
    rec.categoria = categoria
    rec.acao = acao
    rec.justificativa = justificativa
    rec_dict = {"categoria": categoria, "acao": acao, "justificativa": justificativa}
    rec.model_dump.return_value = rec_dict
    rec.dict.return_value = rec_dict
    rec.__getitem__.side_effect = rec_dict.__getitem__ 
    
    mock_parsed.recomendacoes = [rec]
    dict_completo = {
        "pet_id": "1000",
        "status_analise": status,
        "total_recomendacoes": 1,
        "recomendacoes": [rec_dict]
    }
    mock_parsed.model_dump.return_value = dict_completo
    mock_parsed.dict.return_value = dict_completo
    mock_response.parsed = mock_parsed
    
    return mock_response

def test_predict_sem_autenticacao():
    response = client.post("/api/ai/pets/1000/predict", json=VALID_PAYLOAD)
    assert response.status_code == 401

def test_predict_idade_peso_invalido():
    payload_invalido = VALID_PAYLOAD.copy()
    payload_invalido["idade_anos"] = -1.0
    payload_invalido["peso_kg"] = 0.0
    response = client.post("/api/ai/pets/1000/predict", json=payload_invalido, headers=HEADERS)
    assert response.status_code == 422

def test_predict_percentual_acima_100():
    payload_invalido = VALID_PAYLOAD.copy()
    payload_invalido["atividade_diaria_pct"] = 150.0
    response = client.post("/api/ai/pets/1000/predict", json=payload_invalido, headers=HEADERS)
    assert response.status_code == 422

def test_pet_inexistente():
    response = client.get(f"/api/ai/pets/9999/dashboard{QUERY_PARAMS}", headers=HEADERS)
    assert response.status_code in [404, 400]

def test_alteracao_atencao():
    with patch("api.controllers.pet_controller.analysis_service") as mock_analysis:
        mock_analysis.analisar_comportamento.return_value = {
            "status": "ATENÇÃO",
            "mediaHistorica": 65.7,
            "atual": 50.0,
            "variacaoPct": -23.8,
            "justificativaNumerica": "Atividade abaixo do ideal."
        }
        
        response = client.get(f"/api/ai/pets/1000/recommendations{QUERY_PARAMS}", headers=HEADERS)
        assert response.status_code == 200
        assert "ATENÇÃO" in response.text

def test_alteracao_alerta():
    with patch("api.controllers.pet_controller.analysis_service") as mock_analysis:
        mock_analysis.analisar_comportamento.return_value = {
            "status": "ALERTA",
            "mediaHistorica": 65.7,
            "atual": 10.0,
            "variacaoPct": -84.7,
            "justificativaNumerica": "Sinal crítico de letargia."
        }
        
        response = client.get(f"/api/ai/pets/1000/recommendations{QUERY_PARAMS}", headers=HEADERS)
        assert response.status_code == 200
        assert "ALERTA" in response.text
        
        
def test_dashboard_historico_medias():
    resposta_normal = criar_mock_response("NORMAL", "Manutenção", "Monitorar", "Tudo ok")
    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = mock_client_cls.return_value
        mock_instance.models.generate_content.return_value = resposta_normal
        mock_instance.aio.models.generate_content = AsyncMock(return_value=resposta_normal)
        
        response = client.get(f"/api/ai/pets/1000/dashboard{QUERY_PARAMS}", headers=HEADERS)
        assert response.status_code == 200

def test_recommendations_geracao():
    resposta_normal = criar_mock_response("NORMAL", "Manutenção", "Monitorar", "Tudo ok")
    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = mock_client_cls.return_value
        mock_instance.models.generate_content.return_value = resposta_normal
        mock_instance.aio.models.generate_content = AsyncMock(return_value=resposta_normal)
        
        response = client.get(f"/api/ai/pets/1000/recommendations{QUERY_PARAMS}", headers=HEADERS)
        assert response.status_code == 200

def test_machine_learning_predict():

    payload_ml = {
        "idade_anos": 3.5,
        "peso_kg": 15.0,
        "atividade_diaria_pct": 50.0,
        "sono_diario_pct": 60.0,
        "consumo_agua_ml": 400.0
    }
    response = client.post("/api/ai/pets/1000/predict", json=payload_ml, headers=HEADERS)
    assert response.status_code == 200
    assert "predicao" in response.json()

    
def test_ask_pet_dinamico():
    
    with patch("api.controllers.pet_controller.llm_service") as mock_llm:
        mock_llm.answer.return_value = "Resposta personalizada baseada no histórico do pet."
        
        payload = {"pergunta": "Como está a saúde do pet?"}
        response = client.post("/api/ai/pets/1000/ask?pergunta=Como%20está%20a%20saúde%20do%20pet?", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["pet_id"] == "1000"
        assert "resposta" in data

def test_ask_pet_inexistente():
    response = client.post("/api/ai/pets/9999/ask?pergunta=Tudo%20bem?", headers=HEADERS)
    assert response.status_code == 404
    
    
def test_insights_unificado_com_analysis():
    response = client.get("/api/ai/pets/1000/insights", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "status_analise" in data
    assert data["status_analise"] in ["NORMAL", "ATENÇÃO", "ALERTA"]