import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def main():
    data_path = "data/processed/pets_dataset_ready.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Dataset não encontrado em {data_path}. Certifique-se de executar o preprocessing.py.")
        return

    X = df.drop(columns=["status"])
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC()
    }

    best_model_name = ""
    best_model = None
    best_accuracy = 0.0

    print("--- Avaliando Modelos de Machine Learning ---")
    for name, model in models.items():

        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Modelo: {name} | Acurácia: {acc * 100:.2f}%")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    print(f"\n🏆 O melhor modelo foi: {best_model_name} com acurácia de {best_accuracy * 100:.2f}%")

    joblib.dump(best_model, "models/trained/vitalia_rf_model.pkl")
    joblib.dump(scaler, "models/trained/scaler.pkl")
    print("Modelo e scaler salvos com sucesso na pasta models/trained/!")

if __name__ == "__main__":
    main()