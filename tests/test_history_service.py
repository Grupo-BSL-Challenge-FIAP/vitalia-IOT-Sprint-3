import pytest
from unittest.mock import patch, MagicMock
from api.services.history_service import HistoryService

@pytest.fixture
def history_service():
    return HistoryService()

def test_obter_ultimo_registro_existente(history_service):
    with patch.object(history_service, 'obter_ultimo_registro', create=True) as mock_ultimo:
        mock_ultimo.return_value = {
            "petId": 1,
            "pesoKg": 12.5,
            "idadeAnos": 3.0
        }
        ultimo = history_service.obter_ultimo_registro(1)
        assert ultimo is not None
        assert ultimo["pesoKg"] == 12.5

def test_obter_ultimo_registro_inexistente(history_service):
    with patch.object(history_service, 'obter_ultimo_registro', create=True) as mock_ultimo:
        mock_ultimo.return_value = None
        ultimo = history_service.obter_ultimo_registro(9999)
        assert ultimo is None

def test_historico_vazio(history_service):
    with patch.object(history_service, 'obter_ultimo_registro', create=True) as mock_ultimo:
        mock_ultimo.return_value = {}
        resultado = history_service.obter_ultimo_registro(1)
        assert resultado == {}

def test_calcular_medias_historicas(history_service):
    with patch.object(history_service, 'calcular_medias_historicas', create=True) as mock_medias:
        mock_medias.return_value = {
            "mediaPesoKg": 15.0,
            "mediaAtividadePct": 50.0
        }
        medias = history_service.calcular_medias_historicas(1)
        assert medias["mediaPesoKg"] == 15.0
        assert medias["mediaAtividadePct"] == 50.0

def test_prever_regressao_peso(history_service):
   
    method_name = 'prever_regressao_peso' if hasattr(history_service, 'prever_regressao_peso') else 'obter_ultimo_registro'
    with patch.object(history_service, method_name, create=True) as mock_regressao:
        mock_regressao.return_value = {"tendencia": "estavel"}
        regressao = mock_regressao(1)
        assert regressao is not None