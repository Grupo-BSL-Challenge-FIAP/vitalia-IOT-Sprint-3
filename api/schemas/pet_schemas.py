from pydantic import BaseModel

class PetDataInput(BaseModel):
    idade_anos: float
    peso_kg: float
    atividade_diaria_pct: float
    sono_diario_pct: float
    consumo_agua_ml: float

class AskInput(BaseModel):
    pergunta: str

class ReportInput(BaseModel):
    periodo: str  # "diario" ou "semanal"
    dados_atuais: PetDataInput
    historico_recente: list[float] = [15.0, 18.0, 16.0, 15.0]

class DashboardInput(BaseModel):
    historico_atividades: list[float]
    historico_sonos: list[float]
    historico_aguas: list[float]

class RecommendationInput(BaseModel):
    dados_atuais: PetDataInput