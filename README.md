# 🐾 Vitalia AI - API Inteligente de Monitoramento de Saúde Animal

A **Vitalia AI** é uma API de nível de produção desenvolvida em **FastAPI** que une **Machine Learning (Random Forest)**, **Inteligência Artificial Generativa (Google Gemini)** e **Análise de Dados** para monitorar, prever e gerar insights sobre o bem-estar e comportamento de pets.

---

## 🚀 Tecnologias Utilizadas

- **Python 3.14** & **FastAPI**: Construção de endpoints assíncronos de alta performance e documentação interativa (Swagger/OpenAPI).
- **Pydantic V2**: Validação rigorosa de dados de entrada e tipagem estática.
- **Scikit-learn / Joblib**: Classificação baseada em Machine Learning (Random Forest) para predição de status de saúde.
- **Google GenAI SDK (`google-genai`)**: Integração com o Google Gemini para geração de insights humanizados, relatórios veterinários detalhados e sistema de Q&A interativo.
- **Pytest & HTTPX**: Suíte completa de testes automatizados de integração.

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