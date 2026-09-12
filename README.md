# 🐾 Vitalia AI - API de Monitoramento de Saúde Animal

A **Vitalia AI** é uma API desenvolvida em FastAPI que une Machine Learning (Random Forest), Inteligência Artificial Generativa (Google Gemini) e Análise de Dados para monitorar, prever e gerar insights sobre o bem-estar e comportamento de pets.

---

## 👥 Integrantes do Grupo

<table>
  <tr>
    <td width="130">
      <img src="https://github.com/moisesBarsoti.png" width="120" style="border-radius: 50%;"/>
    </td>
    <td>
      <b>Moisés Barsoti Andrade de Oliveira</b><br/>
      <b>RM:</b> 565049 &nbsp;&nbsp;|&nbsp;&nbsp;<b>Turma:</b> 2TDSPO - FIAP <br/>
    </td>
  </tr>

  <tr>
    <td width="130">
      <img src="https://github.com/sSofia-s.png" width="120" style="border-radius: 50%;"/>
    </td>
    <td>
      <b>Sofia Siqueira Fontes</b><br/>
      <b>RM:</b> 563829 &nbsp;&nbsp;|&nbsp;&nbsp;<b>Turma:</b> 2TDSPG - FIAP <br/>
    </td>
  </tr>

  <tr>
    <td width="130">
      <img src="https://github.com/manuelalacerda.png" width="120" style="border-radius: 50%;"/>
    </td>
    <td>
      <b>Manuela de Lacerda Soares</b><br/>
      <b>RM:</b> 564887 &nbsp;&nbsp;|&nbsp;&nbsp;<b>Turma:</b> 2TDSPG - FIAP <br/>
    </td>
  </tr>
</table>

---

## 📋 Documentação e Requisitos do Projeto

### 1. Problema de Negócio
Falta de monitoramento preditivo e automatizado do bem-estar de pets, dificultando a detecção precoce de quedas de atividade e problemas de saúde pelos tutores ou clínicas veterinárias.

### 2. Objetivo da IA
Desenvolver um modelo preditivo capaz de analisar dados de telemetria e hábitos dos animais, gerando alertas precoces e insights explicáveis para otimizar os cuidados veterinários.

### 3. Usuários Beneficiados
Tutores de pets, equipes veterinárias e plataformas de gestão de saúde animal.

### 4. Dados Utilizados & Origem dos Dados
Registros históricos de monitoramento, peso, métricas de atividade diária, padrão de sono e consumo de água gerados por simulações do sistema.

### 5. Informação Destacada sobre IoT Simulado & Formato do Mock
> **⚠️ Aviso Importante sobre IoT:** 
> Atualmente a Vitalia AI **utiliza dados simulados** para representar as informações que futuramente serão coletadas por dispositivos IoT.
> **Formato do Mock de Entrada (Exemplo JSON):**
> ```json
> {
>   "idade_anos": 3.5,
>   "peso_kg": 15.0,
>   "atividade_diaria_pct": 15.0,
>   "sono_diario_pct": 95.0,
>   "consumo_agua_ml": 200.0
> }
> ```

### 6. Criação do Histórico
O histórico e as médias comportamentais dos pets são gerados e consultados através dos serviços internos de dados (`HistoryService`) que mapeiam o ID do pet para simular a telemetria anterior.

### 7. Features Utilizadas
* Atividade diária e histórico de monitoramento.
* Variação de peso (`peso_kg` mapeado para o modelo).
* Padrão de sono e indicadores comportamentais.

### 8. Tratamento dos Dados
Limpeza de valores nulos, normalização de escalas numéricas e mapeamento de features para compatibilidade com o pipeline de machine learning.

### 9. Modelo Selecionado & Justificativa
* **Modelo:** Random Forest (`vitalia_rf_model.pkl` e `scaler.pkl`).
* **Justificativa:** Escolhido pela alta interpretabilidade, eficiência computacional e precisão robusta em dados tabulares de telemetria.

### 10. Métricas Reais & Regressão Simulada
* **Métricas de Classificação:** Validadas com base em acurácia e validação cruzada nos scripts de treino.
* **Regressão de Peso:** Utiliza um regressor linear (`vitalia_peso_regressor.pkl`) para projeção futura baseada em dias simulados.

### 11. Integração com Google Gemini (Gemini)
Integração via Google GenAI SDK (`google-genai`) utilizando o modelo Gemini para traduzir análises de comportamento em recomendações de texto e respostas de Q&A humanizadas.

### 12. Limitações
O modelo depende da consistência dos dados numéricos fornecidos nas requisições; variações drásticas nos inputs podem exigir recalibragem do pipeline.

### 13. Arquitetura
Arquitetura modular em microsserviço estruturada em camadas: `controllers` (rotas FastAPI), `services` (regras de negócio, ML e LLM) e `schemas` (validação com Pydantic).

### 14. Variáveis de Ambiente (.env)
Para rodar a aplicação, crie um arquivo `.env` na raiz com a chave:
```env
GEMINI_API_KEY=sua-chave-aqui
```

### 15. Instalação, Execução & Testes
Instalação das dependências:
```bash
pip install -r requirements.txt
```
Execução da API:
```bash
uvicorn api.main:app --reload
```
Execução da Suíte de Testes (Pytest):
```bash
pytest
```

### 16. Nota sobre as Métricas e Desempenho:
Os resultados obtidos representam o desempenho do modelo sobre dados simulados utilizados para validação técnica do MVP e não representam validação clínica ou desempenho em dados reais.

### 17. Acesso à Documentação (Swagger)
Com a API rodando localmente, acesse:
```bash
http://127.0.0.1:8000/docs
```

## 18. Arquitetura do Sistema

```mermaid
graph TD
    A[Aplicativo] --> B[Backends / APIs]
    B --> C[Banco de dados]
    C --> D[Vitalia AI]
    D --> E[HistoryService]
    E --> F[AnalysisService / ML]
    F --> G[Gemini]
    G --> H[Insights]

    subgraph Evolução Futura
        I[Dispositivo IoT] --> J[Dados reais]
    end
```

### 19. Importante:
No MVP atual, os dados provenientes da camada IoT são simulados. A arquitetura está preparada para substituir essa fonte simulada por dados reais futuramente.
