# pipeline.py

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# ====================
# Carga de Datos
# ====================
def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test

# ====================
# Ingeniería de Características (simple)
# ====================
def feature_engineering(df):
    df = df.copy()
    df['sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['fare'] = pd.qcut(df['Fare'], 13, labels=False, duplicates='drop')
    df['age'] = pd.qcut(df['Age'].fillna(df['Age'].median()), 10, labels=False, duplicates='drop')
    df['pclass'] = df['Pclass']
    df['title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['title'] = df['title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['title'] = df['title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')
    df = pd.get_dummies(df, columns=['title'], drop_first=True)
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    return df

# ====================
# Entrenamiento del modelo
# ====================
def train_model(X, y):
    models = [
        ('mlp', MLPClassifier(max_iter=500)),
        ('rf', RandomForestClassifier(n_estimators=200)),
        ('gbc', GradientBoostingClassifier(n_estimators=200))
    ]
    
    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X, y)
    return ensemble

# ====================
# Evaluación y exportación
# ====================
def evaluate_model(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Curva ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/roc_curve.png")

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.savefig("outputs/figures/confusion_matrix.png")
    print(classification_report(y, y_pred))

# ====================
# Generar archivo de envío
# ====================
def generate_submission(model, test, ids):
    y_pred = model.predict(test)
    submission = pd.DataFrame({"PassengerId": ids, "Survived": y_pred})
    submission.to_csv("data/submission.csv", index=False)
    print("submission.csv generado.")

# ====================
# Main
# ====================
if __name__ == "__main__":
    train, test = load_data()
    
    train_fe = feature_engineering(train)
    test_fe = feature_engineering(test)

    features = ['sex', 'fare', 'age', 'pclass', 'family_size'] + [col for col in train_fe.columns if col.startswith("title_")]

    X_train = train_fe[features]
    y_train = train_fe['Survived']
    X_test = test_fe[features]

    final_model = train_model(X_train, y_train)
    evaluate_model(final_model, X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/best_model.pkl")

    generate_submission(final_model, X_test, test['PassengerId'])
