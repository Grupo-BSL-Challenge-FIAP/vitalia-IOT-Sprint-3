import pytest
from unittest.mock import patch, MagicMock
from api.services.analysis_service import AnalysisService

@pytest.fixture
def analysis_service():
    return AnalysisService()

def test_analise_cenario_normal(analysis_service):
    with patch.object(analysis_service, 'analisar_pet', create=True) as mock_analisar:
        mock_analisar.return_value = {
            "status": "NORMAL",
            "confidence": 0.95,
            "mensagem": "Parâmetros estáveis"
        }
        resultado = analysis_service.analisar_pet(1)
        assert resultado["status"] == "NORMAL"
        assert resultado["confidence"] > 0.9

def test_analise_cenario_attention(analysis_service):
    with patch.object(analysis_service, 'analisar_pet', create=True) as mock_analisar:
        mock_analisar.return_value = {
            "status": "ATTENTION",
            "confidence": 0.82,
            "mensagem": "Atividade ligeiramente baixa"
        }
        resultado = analysis_service.analisar_pet(1)
        assert resultado["status"] == "ATTENTION"

def test_analise_cenario_risk(analysis_service):
    with patch.object(analysis_service, 'analisar_pet', create=True) as mock_analisar:
        mock_analisar.return_value = {
            "status": "RISK",
            "confidence": 0.88,
            "mensagem": "Sinais vitais fora do normal"
        }
        resultado = analysis_service.analisar_pet(1)
        assert resultado["status"] == "RISK"

def test_pipeline_features_e_ordem(analysis_service):
    with patch.object(analysis_service, 'analisar_pet', create=True) as mock_analisar:
        mock_analisar.return_value = {
            "status": "NORMAL",
            "confidence": 0.99,
            "features_utilizadas": ["activity_level", "sleep_hours", "heart_rate_bpm"]
        }
        resultado = mock_analisar(1)
        assert resultado is not None
        assert "features_utilizadas" in resultado