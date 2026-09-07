from pydantic import BaseModel

class PetDataInput(BaseModel):
    idade_anos: float
    peso_kg: float
    atividade_diaria_pct: float
    sono_diario_pct: float
    consumo_agua_ml: float

class QuestionInput(BaseModel):
    pergunta: str
    dados_atuais: PetDataInput 

class ReportInput(BaseModel):
    periodo: str  
    dados_atuais: PetDataInput
    historico_recente: list[float] = [15.0, 18.0, 16.0, 15.0] 

class InsightOutput(BaseModel):
    pet_id: str
    status: str
    mensagem_alerta: str
    explicabilidade: dict 
    insight_ia: str
    
class RecommendationInput(BaseModel):
    dados_atuais: PetDataInput

class DashboardInput(BaseModel):
    historico_atividades: list[float]
    historico_sonos: list[float]
    historico_aguas: list[float]