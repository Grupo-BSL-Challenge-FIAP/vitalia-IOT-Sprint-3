import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

os.environ["API_SECRET_KEY"] = "token-compartilhado-123"
os.environ["GEMINI_API_KEY"] = "fake-key-para-testes"

from api.main import app

client = TestClient(app)
HEADERS = {"Authorization": "Bearer token-compartilhado-123"}
HEADERS_INVALIDO = {"Authorization": "Bearer token-errado"}

VALID_PAYLOAD = {
    "idade_anos": 3.5,
    "peso_kg": 15.0,
    "atividade_diaria_pct": 15.0,
    "sono_diario_pct": 95.0,
    "consumo_agua_ml": 200.0
}

def test_predict_sem_autenticacao():
    response = client.post("/api/ai/pets/1000/predict", json=VALID_PAYLOAD)
    assert response.status_code == 401

def test_token_invalido():
    response = client.post("/api/ai/pets/1000/predict", json=VALID_PAYLOAD, headers=HEADERS_INVALIDO)
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

def test_predict_sono_agua_invalidos():
    payload_invalido = VALID_PAYLOAD.copy()
    payload_invalido["sono_diario_pct"] = 120.0
    payload_invalido["consumo_agua_ml"] = -50.0
    response = client.post("/api/ai/pets/1000/predict", json=payload_invalido, headers=HEADERS)
    assert response.status_code == 422

def test_pet_inexistente():
    response = client.get("/api/ai/pets/9999/dashboard", headers=HEADERS)
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
        
        response = client.get("/api/ai/pets/1000/recommendations", headers=HEADERS)
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
        
        response = client.get("/api/ai/pets/1000/recommendations", headers=HEADERS)
        assert response.status_code == 200
        assert "ALERTA" in response.text

def test_dashboard_endpoint():
    with patch("api.controllers.pet_controller.analysis_service") as mock_analysis:
        mock_analysis.analisar_comportamento.return_value = {
            "status": "NORMAL",
            "mediaHistorica": 60.0,
            "atual": 60.0,
            "variacaoPct": 0.0,
            "justificativaNumerica": "Tudo ok"
        }
        response = client.get("/api/ai/pets/1000/dashboard", headers=HEADERS)
        assert response.status_code == 200

def test_recommendations_endpoint():
    with patch("api.controllers.pet_controller.analysis_service") as mock_analysis:
        mock_analysis.analisar_comportamento.return_value = {
            "status": "NORMAL",
            "mediaHistorica": 60.0,
            "atual": 60.0,
            "variacaoPct": 0.0,
            "justificativaNumerica": "Tudo ok"
        }
        response = client.get("/api/ai/pets/1000/recommendations", headers=HEADERS)
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
    with patch("api.controllers.pet_controller.llm_service") as mock_llm, \
         patch("api.controllers.pet_controller.history_service") as mock_history, \
         patch("api.controllers.pet_controller.analysis_service") as mock_analysis:
        
        mock_history.obter_ultimo_registro.return_value = {"pesoKg": 15.0, "atividadePct": 50.0, "sonoPct": 60.0, "consumoAguaMl": 400.0}
        mock_history.calcular_medias_historicas.return_value = {"mediaPesoKg": 15.0, "mediaAtividadePct": 50.0, "mediaSonoPct": 60.0, "mediaConsumoAguaMl": 400.0}
        mock_analysis.analisar_comportamento.return_value = {"status": "NORMAL", "tendencia": "ESTÁVEL", "alteracoes": []}
        mock_llm.answer.return_value = "Resposta personalizada baseada no histórico do pet."
        
        response = client.post("/api/ai/pets/1000/ask?pergunta=Como%20está%20a%20saúde%20do%20pet?", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["pet_id"] == "1000"
        assert "resposta" in data

def test_ask_pet_inexistente():
    with patch("api.controllers.pet_controller.history_service") as mock_history:
        mock_history.obter_ultimo_registro.return_value = None
        response = client.post("/api/ai/pets/9999/ask?pergunta=Tudo%20bem?", headers=HEADERS)
        assert response.status_code == 404
    
def test_insights_unificado_com_analysis():
    with patch("api.controllers.pet_controller.analysis_service") as mock_analysis, \
         patch("api.controllers.pet_controller.history_service") as mock_history:
        mock_analysis.analisar_comportamento.return_value = {"status": "NORMAL", "justificativaNumerica": "Ok"}
        mock_history.obter_ultimo_registro.return_value = {"pesoKg": 15.0}
        response = client.get("/api/ai/pets/1000/insights", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "status_analise" in data
        assert data["status_analise"] in ["NORMAL", "ATENÇÃO", "ALERTA"]
    
def test_consistencia_status_entre_endpoints():
    with patch("api.controllers.pet_controller.analysis_service") as mock_analysis, \
         patch("api.controllers.pet_controller.history_service") as mock_history, \
         patch("api.controllers.pet_controller.llm_service") as mock_llm:
        
        mock_analysis.analisar_comportamento.return_value = {
            "status": "ATENÇÃO",
            "mediaHistorica": 65.7,
            "atual": 50.0,
            "variacaoPct": -23.8,
            "justificativaNumerica": "Atividade abaixo do ideal.",
            "tendencia": "Estável",
            "alteracoes": ["Queda leve na atividade"]
        }
        
        mock_history.obter_ultimo_registro.return_value = {
            "pesoKg": 15.0, "atividadePct": 40.0, "sonoPct": 80.0, "consumoAguaMl": 300.0
        }
        mock_history.calcular_medias_historicas.return_value = {
            "mediaPesoKg": 15.0, "mediaAtividadePct": 50.0, "mediaSonoPct": 80.0, "mediaConsumoAguaMl": 300.0
        }
        mock_history.obter_historico_pet.return_value = [
            {"data": "2026-01-01", "pesoKg": 15.0},
            {"data": "2026-01-02", "pesoKg": 15.1}
        ]
        
        mock_llm.answer.return_value = "Resposta de teste"

        resp_dashboard = client.get("/api/ai/pets/1000/dashboard", headers=HEADERS)
        resp_insights = client.get("/api/ai/pets/1000/insights", headers=HEADERS)
        resp_recommendations = client.get("/api/ai/pets/1000/recommendations", headers=HEADERS)
        resp_trends = client.get("/api/ai/pets/1000/trends", headers=HEADERS)
        resp_ask = client.post("/api/ai/pets/1000/ask?pergunta=Status?", headers=HEADERS)

        assert resp_dashboard.status_code == 200
        assert resp_insights.status_code == 200
        assert resp_recommendations.status_code == 200
        assert resp_trends.status_code == 200
        assert resp_ask.status_code == 200

        status_dashboard = resp_dashboard.json().get("status_geral")
        status_insights = resp_insights.json().get("status_analise")
        status_recommendations = resp_recommendations.json().get("status_analise")

        assert status_dashboard == "ATENÇÃO"
        assert status_insights == "ATENÇÃO"
        assert status_recommendations == "ATENÇÃO"
        assert status_dashboard == status_insights == status_recommendations

        mock_llm.answer.assert_called_once()
        args, _ = mock_llm.answer.call_args
        contexto_enviado = args[0]
        assert contexto_enviado.get("classificacao") == "ATENÇÃO"