# 🐾 Vitalia AI - API Inteligente de Monitoramento de Saúde Animal

A **Vitalia AI** é uma API de nível de produção desenvolvida em FastAPI que une Machine Learning (Random Forest), Inteligência Artificial Generativa (Google Gemini) e Análise de Dados para monitorar, prever e gerar insights sobre o bem-estar e comportamento de pets.

---

## 🚀 Tecnologias Utilizadas

* **Python 3.14 & FastAPI**: Construção de endpoints assíncronos de alta performance e documentação interativa (Swagger/OpenAPI).
* **Pydantic V2**: Validação rigorosa de dados de entrada e tipagem estática.
* **Scikit-learn / Joblib**: Classificação baseada em Machine Learning (Random Forest) para predição de status de saúde.
* **Google GenAI SDK**: Integração com o Google Gemini para geração de insights humanizados, relatórios veterinários detalhados e sistema de Q&A interativo.
* **Pytest & HTTPX**: Suíte completa de testes automatizados de integração.

---

## 🔗 Acesso à Documentação (Swagger)

Com a API em execução localmente, acesse a documentação interativa e os endpoints de teste através do link:
* **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 📂 Estrutura do Projeto

```text
vitalia-ia/
│
├── api/
│   └── main.py              # Arquivo principal com todas as rotas e regras de negócio da API
├── models/
│   ├── pet_health_model.pkl # Modelo treinado de Machine Learning (Random Forest)
│   └── scaler.pkl           # Escalador de features para normalização dos dados
├── tests/
│   └── test_api.py          # Suíte de testes automatizados (Pytest)
├── requirements.txt         # Dependências do projeto
└── README.md                # Documentação oficial da API

```

## 📋 Informações do Projeto
### Problema de negócio 
Falta de monitoramento preditivo e automatizado do bem-estar de pets, dificultando a detecção precoce de quedas de atividade e problemas de saúde pelos tutores ou clínicas veterinárias.

### Objetivo da IA 
Desenvolver um modelo preditivo capaz de analisar dados de telemetria e hábitos dos animais, gerando alertas precoces e insights explicáveis para otimizar os cuidados veterinários.

### Usuários beneficiados 
Tutores de pets, equipes veterinárias e plataformas de gestão de saúde animal (como o ecossistema Vitalia).

### Dados utilizados & Origem dos dados 
Registros históricos de monitoramento IoT, peso, métricas de atividade diária, padrão de sono e consumo de água gerados por dispositivos conectados e simulações do sistema.

### Tratamento dos dados
Limpeza de valores nulos, normalização de escalas numéricas e mapeamento/conversão de features para compatibilidade com o pipeline de machine learning.

### Features utilizadas:

* Atividade diária e histórico de monitoramento IoT.

* Variação de peso (peso_kg mapeado para o modelo).

* Padrão de sono e indicadores comportamentais.

### Algoritmo escolhido & Justificativa
Modelos de Machine Learning baseados em classificação e regressão (como Random Forest / Regressão Linear), escolhidos pela alta interpretabilidade, eficiência computacional e precisão em dados tabulares de telemetria.

### Treinamento, Métricas & Resultados
Treinamento realizado via scripts dedicados com validação cruzada. Métricas de regressão/classificação validadas (como acurácia, erro e matriz de confusão) demonstrando alta confiabilidade nas previsões.

### Limitações 
O modelo depende da regularidade dos dados coletados pelos dispositivos IoT; variações drásticas na captura de sensores podem requerer recalibragem periódica.

### Arquitetura, Fluxo de integração & Estratégia de deploy
Arquitetura em microsserviço com FastAPI (api/main.py), conteinerizada via Docker, protegida por chave de API (x-api-key) e variáveis de ambiente (.env), integrando o Frontend/Backend diretamente ao motor de IA.
