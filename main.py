# main.py

import pandas as pd
import joblib
import os

# ==========================
# Cargar modelo entrenado
# ==========================
def load_model(model_path="models/best_model.pkl"):
    return joblib.load(model_path)

# ==========================
# Cargar y preparar datos de prueba
# ==========================
def load_test_data(path="data/test.csv"):
    df = pd.read_csv(path)
    df['sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['fare'] = pd.qcut(df['Fare'], 13, labels=False, duplicates='drop')
    df['age'] = pd.qcut(df['Age'].fillna(df['Age'].median()), 10, labels=False, duplicates='drop')
    df['pclass'] = df['Pclass']
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['title_Mr'] = 1  # ajuste para compatibilidad con features esperados
    return df

# ==========================
# Predecir y exportar resultados
# ==========================
def predict_and_export(model, test_df, features, output_path="data/submission.csv"):
    y_pred = model.predict(test_df[features])
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_pred
    })
    submission.to_csv(output_path, index=False)
    print(f"Archivo guardado en: {output_path}")

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    model = load_model()
    test_df = load_test_data()

    features = ['sex', 'fare', 'age', 'pclass', 'family_size', 'title_Mr']

    predict_and_export(model, test_df, features)