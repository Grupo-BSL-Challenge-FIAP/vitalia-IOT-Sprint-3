import os
from google import genai

class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def answer(self, contexto: dict, pergunta: str) -> str:
        prompt = f"""
Você é o assistente virtual da Vitalia AI. A Vitalia AI é um apoio ao acompanhamento e à decisão, não substituição do veterinário.

Diretrizes obrigatórias:
- Não invente informações.
- Não realize diagnóstico.
- Não afirme que o animal possui uma doença.
- Não prescreva medicamentos ou tratamentos.
- Oriente avaliação veterinária quando necessário.

Contexto estruturado do pet:
{contexto}

Pergunta do tutor:
{pergunta}
"""
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

llm_service = LLMService()
