# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =====================
# Configuración inicial
# =====================
st.set_page_config(page_title="Predicción de Supervivencia - Titanic", layout="wide")
st.title("🚢 Predicción de Supervivencia en el Titanic")
st.markdown("Sube un archivo `.csv` con los datos de pasajeros para predecir la supervivencia utilizando un modelo entrenado.")

# =====================
# Cargar modelo
# =====================
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

# =====================
# Procesamiento de datos
# =====================
def preprocess(df):
    df = df.copy()

    # Imputación de edad por grupo (Pclass y Sex)
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    df['sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['fare'] = pd.qcut(df['Fare'], 13, labels=False, duplicates='drop')
    df['age'] = pd.qcut(df['Age'], 10, labels=False, duplicates='drop')
    df['pclass'] = df['Pclass']
    df['family_size'] = df['SibSp'] + df['Parch'] + 1

    # Extraer y agrupar 'title'
    df['title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['title'] = df['title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['title'] = df['title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')

    # One-hot encode con columnas esperadas
    dummies = pd.get_dummies(df['title'], prefix='title')
    for col in ['title_Miss', 'title_Mr', 'title_Mrs', 'title_Rare']:
        if col not in dummies:
            dummies[col] = 0
    df = pd.concat([df, dummies], axis=1)

    return df

# =====================
# Subida del archivo
# =====================
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV de prueba", type=["csv"])

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        st.success("Archivo cargado correctamente ✅")

        processed_df = preprocess(test_df)
        features = ['sex', 'fare', 'age', 'pclass', 'family_size',
                    'title_Miss', 'title_Mr', 'title_Mrs', 'title_Rare']

        predictions = model.predict(processed_df[features])
        test_df['Survived'] = predictions

        st.dataframe(test_df[['PassengerId', 'Survived']].head())

        csv = test_df[['PassengerId', 'Survived']].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar submission.csv", data=csv, file_name="submission.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
else:
    st.info("Esperando archivo CSV...")