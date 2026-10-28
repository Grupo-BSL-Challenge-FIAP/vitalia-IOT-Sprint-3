import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

load_dotenv()

class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def answer(self, contexto: str, pergunta: str) -> str:
        prompt = f"""
Você é o assistente virtual da Vitalia AI. A Vitalia AI é um apoio ao acompanhamento e à decisão.

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
        max_retries = 3
        for tentativa in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )
                return response.text
            except ServerError as e:
                if tentativa == max_retries - 1:
                    raise e
                time.sleep(2) 
llm_service = LLMService()