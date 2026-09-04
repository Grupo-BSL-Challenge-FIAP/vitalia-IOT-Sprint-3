import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def executar_treinamento():
    print("1. Criando dataset simulado...")
    np.random.seed(42)
    n_samples = 200
    df = pd.DataFrame({
        "atividade": np.random.uniform(30, 100, n_samples),
        "peso_var": np.random.uniform(-5, 5, n_samples),
        "sono": np.random.choice([0, 1], n_samples),
        "alerta": np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    X = df[["atividade", "peso_var", "sono"]]
    y = df["alerta"]
    
    print("2. Separando dados de treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("3. Treinando modelos (Random Forest e Logistic Regression)...")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.log_loss = None # placeholder
    lr_model.fit(X_train, y_train)
    
    print("4. Comparando modelos e avaliando resultados...")
    rf_pred = rf_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    
    rf_acc = accuracy_score(y_test, rf_pred)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    print(f"Acurácia Random Forest: {rf_acc:.2f}")
    print(f"Acurácia Logistic Regression: {lr_acc:.2f}")
    
    print("5. Selecionando o melhor modelo...")
    melhor_modelo = rf_model if rf_acc >= lr_acc else lr_model
    nome_melhor = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
    print(f"Modelo selecionado: {nome_melhor}")
    
    print("6. Gerando gráfico de avaliação (Matplotlib)...")
    plt.figure(figsize=(6, 4))
    plt.bar(["Random Forest", "Logistic Regression"], [rf_acc, lr_acc], color=["#4CAF50", "#2196F3"])
    plt.ylabel("Acurácia")
    plt.title("Comparação de Modelos - Vitalia AI")
    plt.savefig("models/trained/comparacao_modelos.png")
    plt.close()
    
    print("7. Salvando modelo treinado...")
    joblib.dump(melhor_modelo, "models/trained/vitalia_rf_model.pkl")
    print("Modelo salvo com sucesso em models/trained/vitalia_rf_model.pkl!")

if __name__ == "__main__":
    executar_treinamento()