from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

class PetDataInput(BaseModel):
    idade_anos: float = Field(ge=0, le=40)
    peso_kg: float = Field(gt=0)
    atividade_diaria_pct: float = Field(ge=0, le=100)
    sono_diario_pct: float = Field(ge=0, le=100)
    consumo_agua_ml: float = Field(ge=0)
    historico_medicoes: Optional[List[float]] = Field(default=None)

    @field_validator('idade_anos', 'peso_kg', 'atividade_diaria_pct', 'sono_diario_pct', 'consumo_agua_ml', mode='before')
    def validar_tipos_invalidos(cls, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Tipo inválido. O valor deve ser um número numérico (inteiro ou flutuante).")
        return float(value)

    @model_validator(mode='after')
    def validar_historico(self):
        if self.historico_medicoes is not None and len(self.historico_medicoes) == 0:
            raise ValueError("O histórico de medições não pode estar vazio.")
        
        if self.historico_medicoes is not None and len(self.historico_medicoes) > 100:
            raise ValueError("Tamanho incompatível. O histórico excede o limite máximo de 100 registros.")
            
        return self

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