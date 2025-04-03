# 🚢 Titanic - Predicción de Supervivencia

Este repositorio contiene un análisis completo y profesional del famoso conjunto de datos del Titanic (Kaggle). Se construye un pipeline de ciencia de datos que incluye desde el análisis exploratorio hasta la implementación de modelos avanzados y ensamblado.

## 📂 Estructura del Proyecto

- `notebooks/`: notebook principal con el análisis completo (`titanic_analysis.ipynb`)
- `scripts/`: script Python del pipeline (opcional)
- `data/`: datasets originales y archivo de predicciones
- `models/`: modelos entrenados exportados (`.pkl`)
- `outputs/`: gráficas, tablas de resultados y figuras
- `requirements.txt`: dependencias del proyecto

## 🧪 Contenido del Análisis

- Análisis exploratorio detallado (EDA)
- Ingeniería de características avanzada
- Selección de variables (correlaciones, chi-cuadrado, MI)
- Transformaciones (Label, Ordinal, OneHot)
- Entrenamiento de modelos y comparación
- Voting y Stacking Ensemble
- Curvas ROC, AUC, matrices de confusión
- Exportación de predicciones para Kaggle

## 🛠️ Requisitos

```bash
pip install -r requirements.txt
```

## 📈 Modelo Final

El mejor modelo fue un **VotingClassifier** compuesto por:
- MLPClassifier
- RandomForestClassifier
- GradientBoostingClassifier

## 🧠 Resultado

Evaluado mediante validación cruzada, AUC y matriz de confusión.

## 📤 Exportación

El archivo `submission.csv` fue generado para participar en la competencia de Kaggle.

## 🚀 Cómo ejecutar el proyecto

1. Clona este repositorio:

```bash
git clone https://github.com/tu_usuario/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Coloca los archivos `train.csv` y `test.csv` dentro de la carpeta `data/`.

4. Ejecuta el pipeline completo:

```bash
python scripts/pipeline.py
```

Esto generará:

- `models/best_model.pkl`: modelo final entrenado
- `data/submission.csv`: predicciones listas para Kaggle
- Figuras en `outputs/figures/`
