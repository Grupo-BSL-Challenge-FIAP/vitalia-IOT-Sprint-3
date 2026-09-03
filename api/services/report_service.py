import pandas as pd
from datetime import datetime

class ReportService:
    @staticmethod
    def gerar_relatorio(pet_id: int, tipo_periodo: str = "mensal"):
        periodos_map = {
            "diario": "Últimas 24 horas",
            "semanal": "Últimos 7 dias",
            "mensal": "01/08/2026 – 31/08/2026"
        }
        
        perimetro = periodos_map.get(tipo_periodo, "Período personalizado")
        
        indicadores = {
            "atividade_variacao": "+12%",
            "sono_status": "Estável",
            "peso_variacao": "-2%"
        }
        
        dados_graficos = {
            "labels": ["Semana 1", "Semana 2", "Semana 3", "Semana 4"],
            "atividade": [75, 82, 78, 85],
            "sono": [65, 68, 70, 67],
            "peso": [7.1, 7.0, 6.9, 6.9]
        }
        
        alertas_count = 2
        resumo = (
            f"O pet apresentou variação controlada e redução leve de atividade nos últimos dias. "
            f"Foram detectadas {alertas_count} ocorrências de atenção no período."
        )
        
        return {
            "pet_id": pet_id,
            "tipo_relatorio": tipo_periodo,
            "periodo": perimetro,
            "data_emissao": datetime.now().strftime("%d/%m/%Y"),
            "indicadores": indicadores,
            "graficos": dados_graficos,  
            "alertas_identificados": alertas_count,
            "resumo_automatico": resumo,
            "status_exportacao": "Pronto para download (JSON/PDF)" 
        }