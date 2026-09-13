# ðŸ¾ Vitalia AI - API de Monitoramento de SaÃºde Animal

A **Vitalia AI** Ã© uma API desenvolvida em FastAPI que une Machine Learning (Random Forest), InteligÃªncia Artificial Generativa (Google Gemini) e AnÃ¡lise de Dados para monitorar, prever e gerar insights sobre o bem-estar e comportamento de pets.

---

## ðŸ‘¥ Integrantes do Grupo

<table>
  <tr>
    <td width="130">
      <img src="https://github.com/moisesBarsoti.png" width="120" style="border-radius: 50%;"/>
    </td>
    <td>
      <b>MoisÃ©s Barsoti Andrade de Oliveira</b><br/>
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

## ðŸ“‹ DocumentaÃ§Ã£o e Requisitos do Projeto

### 1. Problema de NegÃ³cio
Falta de monitoramento preditivo e automatizado do bem-estar de pets, dificultando a detecÃ§Ã£o precoce de quedas de atividade e problemas de saÃºde pelos tutores ou clÃ­nicas veterinÃ¡rias.

### 2. Objetivo da IA
Desenvolver um modelo preditivo capaz de analisar dados de telemetria e hÃ¡bitos dos animais, gerando alertas precoces e insights explicÃ¡veis para otimizar os cuidados veterinÃ¡rios.

### 3. UsuÃ¡rios Beneficiados
Tutores de pets, equipes veterinÃ¡rias e plataformas de gestÃ£o de saÃºde animal.

### 4. Dados Utilizados & Origem dos Dados
Registros histÃ³ricos de monitoramento, peso, mÃ©tricas de atividade diÃ¡ria, padrÃ£o de sono e consumo de Ã¡gua gerados por simulaÃ§Ãµes do sistema.

### 5. InformaÃ§Ã£o Destacada sobre IoT Simulado & Formato do Mock
> **âš ï¸ Aviso Importante sobre IoT:** 
> Atualmente a Vitalia AI **utiliza dados simulados** para representar as informaÃ§Ãµes que futuramente serÃ£o coletadas por dispositivos IoT.
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

### 6. CriaÃ§Ã£o do HistÃ³rico
O histÃ³rico e as mÃ©dias comportamentais dos pets sÃ£o gerados e consultados atravÃ©s dos serviÃ§os internos de dados (`HistoryService`) que mapeiam o ID do pet para simular a telemetria anterior.

### 7. Features Utilizadas
* Atividade diÃ¡ria e histÃ³rico de monitoramento.
* VariaÃ§Ã£o de peso (`peso_kg` mapeado para o modelo).
* PadrÃ£o de sono e indicadores comportamentais.

### 8. Tratamento dos Dados
Limpeza de valores nulos, normalizaÃ§Ã£o de escalas numÃ©ricas e mapeamento de features para compatibilidade com o pipeline de machine learning.

### 9. Modelo Selecionado & Justificativa
* **Modelo:** Random Forest (`vitalia_rf_model.pkl` e `scaler.pkl`).
* **Justificativa:** Escolhido pela alta interpretabilidade, eficiÃªncia computacional e precisÃ£o robusta em dados tabulares de telemetria.

### 10. Métricas Reais
* **Métricas de Classificação:** Validadas com base em acurácia e validação cruzada nos scripts de treino do modelo Random Forest.

### 11. IntegraÃ§Ã£o com Google Gemini (Gemini)
IntegraÃ§Ã£o via Google GenAI SDK (`google-genai`) utilizando o modelo Gemini para traduzir anÃ¡lises de comportamento em recomendaÃ§Ãµes de texto e respostas de Q&A humanizadas.

### 12. LimitaÃ§Ãµes
O modelo depende da consistÃªncia dos dados numÃ©ricos fornecidos nas requisiÃ§Ãµes; variaÃ§Ãµes drÃ¡sticas nos inputs podem exigir recalibragem do pipeline.

### 13. Arquitetura
Arquitetura modular em microsserviÃ§o estruturada em camadas: `controllers` (rotas FastAPI), `services` (regras de negÃ³cio, ML e LLM) e `schemas` (validaÃ§Ã£o com Pydantic).

### 14. VariÃ¡veis de Ambiente (.env)
Para rodar a aplicaÃ§Ã£o, crie um arquivo `.env` na raiz com a chave:
```env
GEMINI_API_KEY=sua-chave-aqui
```

### 15. InstalaÃ§Ã£o, ExecuÃ§Ã£o & Testes
InstalaÃ§Ã£o das dependÃªncias:
```bash
pip install -r requirements.txt
```
ExecuÃ§Ã£o da API:
```bash
uvicorn api.main:app --reload
```
ExecuÃ§Ã£o da SuÃ­te de Testes (Pytest):
```bash
pytest
```

### 16. Nota sobre as MÃ©tricas e Desempenho:
Os resultados obtidos representam o desempenho do modelo sobre dados simulados utilizados para validaÃ§Ã£o tÃ©cnica do MVP e nÃ£o representam validaÃ§Ã£o clÃ­nica ou desempenho em dados reais.

### 17. Acesso Ã  DocumentaÃ§Ã£o (Swagger)
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

    subgraph EvoluÃ§Ã£o Futura
        I[Dispositivo IoT] --> J[Dados reais]
    end
```

### 19. Importante:
No MVP atual, os dados provenientes da camada IoT sÃ£o simulados. A arquitetura estÃ¡ preparada para substituir essa fonte simulada por dados reais futuramente.

## ðŸ“Š Fluxo de Dados: Atual vs. Arquitetura Futura

### MVP Atual
No estÃ¡gio atual do MVP, o fluxo de dados opera de forma simulada e integrada aos serviÃ§os internos:
```text
Dados simulados â†’ HistoryService â†’ AnalysisService â†’ Machine Learning â†’ Gemini â†’ FastAPI â†’ Aplicativo
```

### 20. Arquitetura Futura:
Na evoluÃ§Ã£o planejada para o projeto, a coleta de dados passarÃ¡ a ser automatizada por hardware real conectando-se diretamente ao ecossistema de persistÃªncia e inteligÃªncia:
```text
Dispositivo IoT â†’ Backend â†’ Oracle â†’ IA â†’ Aplicativo
```

### 21. âš ï¸ AtenÃ§Ã£o
A integraÃ§Ã£o fÃ­sica e automatizada (ESP32 â†’ Oracle â†’ IA) representa a evoluÃ§Ã£o futura e ainda nÃ£o estÃ¡ ativa no ambiente do MVP atual.






