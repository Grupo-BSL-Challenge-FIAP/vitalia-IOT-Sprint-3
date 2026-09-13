import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def train_and_save_pipeline(processed_filepath: str, model_output_path: str):
    print("Iniciando carregamento e comparacao de modelos candidatos...")
    df = pd.read_csv(processed_filepath)
    
    features = ["idade_anos", "peso_kg", "atividade_diaria_pct", "sono_diario_pct", "consumo_agua_ml"]
    X = df[features]
    y = df['status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    candidates = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(random_state=42)
    }
    
    best_score = -1
    best_name = None
    best_model = None
    scores = {}
    
    print("\nAvaliando candidatos (métrica: F1-Macro):")
    for name, model in candidates.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = f1_score(y_test, y_pred, average="macro", zero_division=0)
        scores[name] = score
        print(f"-> {name}: {score:.4f}")
        
        if score > best_score or (score == best_score and name == "Random Forest"):
            best_score = score
            best_name = name
            best_model = pipeline

    print(f"\n[OK] Melhor modelo selecionado: {best_name} com F1-Macro de {best_score:.4f}")
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_model, model_output_path)
    print(f"[OK] Pipeline salvo com sucesso em: {model_output_path}")
            
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs("models", exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette="Blues_d")
    plt.ylim(0, 1.05)
    plt.ylabel("F1-Score (Macro)")
    plt.title("Comparação de Modelos - Vitalia AI")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("models/model_comparison.png")
    plt.close()
    print("[OK] Gráfico de comparação salvo em: models/model_comparison.png")
    
if __name__ == "__main__":
    train_and_save_pipeline(
        "data/processed/pets_dataset_ready.csv",
        "models/trained/vitalia_pipeline.pkl"
    )
