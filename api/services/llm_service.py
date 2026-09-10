import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não encontrada nas variáveis de ambiente ou arquivo .env.")
        self.client = genai.Client(api_key=api_key)

    def answer(self, context: dict, question: str) -> str:
        prompt = f"""
        Com base estritamente nos dados estruturados do pet abaixo, responda à pergunta do usuário.
        Não invente informações que não estejam presentes no contexto.

        --- CONTEXTO ESTRUTURADO ---
        - Identificação: {context.get('identificacao')}
        - Dados Atuais: {context.get('dados_atuais')}
        - Médias Históricas: {context.get('medias_historicas')}
        - Tendência: {context.get('tendencia')}
        - Classificação ML: {context.get('classificacao_ml')}
        - Alterações Encontradas: {context.get('alteracoes_encontradas')}
        ----------------------------

        Pergunta: {question}
        """

        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text

llm_service = LLMService()