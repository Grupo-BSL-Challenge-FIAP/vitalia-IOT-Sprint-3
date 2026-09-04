import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def executar_avaliacao():
    print("1. Carregando dados para avaliação...")
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    print("2. Calculando métricas de classificação...")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"Confusion Matrix:\n{cm}")
    
    print("3. Gerando visualizações (Matriz de confusão e gráficos)...")
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão - Vitalia AI")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("models/trained/matriz_confusao.png")
    plt.close()
    
    print("Avaliação concluída com sucesso!")

if __name__ == "__main__":
    executar_avaliacao()