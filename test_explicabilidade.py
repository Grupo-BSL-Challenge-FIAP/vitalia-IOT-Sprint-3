def gerar_explicacao_ia(pet_id: str, dados_atuais: dict, dados_historicos: dict):
    
    reducao_atividade = ((dados_historicos["media_atividade"] - dados_atuais["atividade"]) / dados_historicos["media_atividade"]) * 100
    
    resposta = f"""
    [Atenção]
    A atividade do pet {pet_id} apresentou redução de {reducao_atividade:.0f}% nos últimos {dados_atuais['periodo_dias']} dias.
    
    Motivo:
    A média atual de atividade física diária ({dados_atuais['atividade']} min) está significativamente abaixo do padrão histórico do pet ({dados_historicos['media_atividade']} min).
    
    Dados utilizados: Atividade diária, histórico de monitoramento IoT, peso ({dados_atuais['peso']}kg).
    Período analisado: Últimos {dados_atuais['periodo_dias']} dias.
    Evidência numérica: Queda de {reducao_atividade:.1f}% em relação à linha de base.
    """
    return resposta.strip()

if __name__ == "__main__":
    pet_exemplo = "thor-123"
    atual = {"atividade": 28, "peso": 10.5, "periodo_dias": 7}
    historico = {"media_atividade": 39}
    
    print(gerar_explicacao_ia(pet_exemplo, atual, historico))