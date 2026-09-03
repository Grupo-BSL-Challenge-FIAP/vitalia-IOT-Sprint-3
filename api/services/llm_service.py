from google import genai
import os

# Configura a chave diretamente no ambiente para o SDK novo reconhecer
os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6L7rbTiIWKBurqyrKghqRoN8xqDAfDi4I4wGDmCrTvOig"

def generate_pet_insight(dados_pet: dict, status_ml: str) -> str:
    try:
        # Inicializa o client sem parâmetros, pois ele lê do os.environ
        client = genai.Client()
        
        prompt = f"""
        Você é a Vitalia AI, uma assistente virtual de acompanhamento pet.
        Explique os dados deste pet para o tutor de forma amigável e objetiva, sem diagnósticos médicos.
        
        Dados:
        - Idade: {dados_pet['idade_anos']} anos
        - Peso: {dados_pet['peso_kg']} kg
        - Atividade: {dados_pet['atividade_diaria_pct']}%
        - Sono: {dados_pet['sono_diario_pct']}%
        - Água: {dados_pet['consumo_agua_ml']} ml
        - Status do modelo: {status_ml}
        
        Escreva um parágrafo curto de até 3 frases.
        """
        
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return response.text.strip()
        
    except Exception as e:
        print(f"ERRO REAL DO GEMINI: {str(e)}")
        if status_ml == "NORMAL":
            return "O comportamento do pet está dentro do padrão histórico normal."
        return "Notamos variações nos padrões do pet. Recomendamos atenção aos dados."