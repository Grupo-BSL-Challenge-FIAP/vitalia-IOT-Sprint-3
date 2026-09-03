from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "idade_anos": 3.5,
        "peso_kg": 15.0,
        "atividade_diaria_pct": 15.0,
        "sono_diario_pct": 95.0,
        "consumo_agua_ml": 200.0
    }
    response = client.post("/api/ai/pets/thor-123/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["pet_id"] == "thor-123"
    assert "status" in data
    assert "explicabilidade" in data
    assert "insight_ia" in data

def test_recommendations_endpoint():
    payload = {
        "dados_atuais": {
            "idade_anos": 3.5,
            "peso_kg": 15.0,
            "atividade_diaria_pct": 15.0,
            "sono_diario_pct": 95.0,
            "consumo_agua_ml": 200.0
        }
    }
    response = client.post("/api/ai/pets/thor-123/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["pet_id"] == "thor-123"
    assert data["total_recomendacoes"] > 0
    assert isinstance(data["recomendacoes"], list)

def test_dashboard_endpoint():
    payload = {
        "historico_atividades": [15.0, 20.0, 18.0],
        "historico_sonos": [95.0, 90.0, 92.0],
        "historico_aguas": [200.0, 250.0, 220.0]
    }
    response = client.post("/api/ai/pets/thor-123/dashboard", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["pet_id"] == "thor-123"
    assert "indicadores_consolidados" in data
    assert data["indicadores_consolidados"]["total_registros_analisados"] == 3