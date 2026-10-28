import pytest
from unittest.mock import MagicMock, patch
from api.services.llm_service import LLMService

@pytest.fixture
def llm_service_mocked():
    """Fixture que inicializa o LLMService com uma chave simulada."""
    with patch("api.services.llm_service.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        service = LLMService()
        service.client = mock_client
        return service, mock_client

def test_funcionamento_normal_do_gemini(llm_service_mocked):
    """Testa o funcionamento normal do Gemini (Mockado)."""
    service, mock_client = llm_service_mocked
    
    mock_response = MagicMock()
    mock_response.text = "Resumo analítico gerado pela IA: Pet estável e normal."
    mock_client.models.generate_content.return_value = mock_response

    resultado = service.answer("Contexto: Pet Normal", "Como está?")
    
    assert "Resumo analítico gerado pela IA" in resultado
    mock_client.models.generate_content.assert_called_once()

def test_fallback_do_gemini_sem_cliente():
    """Testa o fallback quando a API Key está ausente (cliente não inicializado)."""
    with patch("api.services.llm_service.os.getenv", return_value=None):
        service = LLMService()
        resultado = service.answer("Contexto do pet", "Pergunta?")
        
        assert "[Modo Fallback - IA Indisponível/Sem Chave]" in resultado
        assert "Contexto do pet" in resultado

def test_fallback_do_gemini_por_erro_servidor(llm_service_mocked):
    """Testa o fallback quando a API do Gemini retorna erro de servidor."""
    service, mock_client = llm_service_mocked

    mock_client.models.generate_content.side_effect = Exception("Erro Servidor Indisponível")

    resultado = service.answer("Contexto de Risco", "O que se passa?")
    
    assert "[Modo Fallback de Segurança]" in resultado
    assert "Contexto de Risco" in resultado

@pytest.mark.parametrize("estado", ["NORMAL", "ATTENTION", "RISK"])
def test_cenarios_pet(llm_service_mocked, estado):
    """Testa a geração de resposta para diferentes cenários do pet."""
    service, mock_client = llm_service_mocked
    
    mock_response = MagicMock()
    mock_response.text = f"Análise para o estado: {estado}"
    mock_client.models.generate_content.return_value = mock_response

    resultado = service.answer(f"Estado atual: {estado}", "Avaliar estado")
    assert estado in resultado