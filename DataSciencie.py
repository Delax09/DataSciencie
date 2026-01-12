#Esto es para aprender y estudiar Data Science y tener algun conocimiento 
#en el mundo laboral y/o buscar practicas de Cientifico de datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Cargar el dataset
dataSet = pd.read_csv('spotify-2023.csv', encoding='latin-1')

print("=== ANÁLISIS DE POPULARIDAD - SPOTIFY 2023 ===\n")

# 1. INFORMACIÓN GENERAL
print("1. INFORMACIÓN GENERAL DEL DATASET")
print(f"Número de canciones: {len(dataSet)}")
print(f"Número de columnas: {len(dataSet.columns)}")

# 3. VALORES NULOS
print("\n3. VALORES NULOS POR COLUMNA")
print(dataSet.isnull().sum())

# 4. LIMPIEZA DE DATOS
print("\n4. LIMPIEZA DE DATOS")
# Convertir streams a numérico
dataSet['streams'] = pd.to_numeric(dataSet['streams'], errors='coerce')

# Eliminar filas con valores nulos en columnas clave
dataSet_clean = dataSet.dropna(subset=['streams', 'danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'bpm'])
print(f"Filas después de limpieza: {len(dataSet_clean)}")

# 5. CORRELACIONES
print("\n5. CORRELACIONES CON STREAMS")
columnas_numericas = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'bpm', 'in_spotify_playlists', 'in_spotify_charts']
correlaciones = dataSet_clean[columnas_numericas + ['streams']].corr()['streams'].sort_values(ascending=False)


# 6. MODELO DE REGRESIÓN
print("\n6. MODELO DE PREDICCIÓN DE STREAMS")
X = dataSet_clean[columnas_numericas]
y = dataSet_clean['streams']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Métricas
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.0f}")
print(f"\nCoeficientes del modelo:")
for col, coef in zip(columnas_numericas, modelo.coef_):
    print(f"  {col}: {coef:.2e}")

# 7. TOP 10 CANCIONES MÁS POPULARES
print("\n7. TOP 10 CANCIONES MÁS POPULARES")
top_10 = dataSet_clean.nlargest(10, 'streams')[['track_name', 'artist(s)_name', 'streams']]
print(top_10.to_string())


# 8. CLASIFICACIÓN: HIT vs NO HIT
print("\n8. CLASIFICACIÓN: HIT vs NO HIT")
# Definir un "Hit" como una canción que está por encima de la mediana de streams
mediana_streams = dataSet_clean['streams'].median()
dataSet_clean['is_hit'] = (dataSet_clean['streams'] > mediana_streams).astype(int)
print(f"Mediana de streams: {mediana_streams:.0f}")
print(f"Hits (1): {(dataSet_clean['is_hit'] == 1).sum()} canciones")

# Preparar datos para clasificación
X = dataSet_clean[columnas_numericas]
y = dataSet_clean['is_hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODELO 1: Decision Tree
print("\n--- DECISION TREE ---")
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt):.4f}")

# MODELO 2: Random Forest
print("\n--- RANDOM FOREST ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")

# MODELO 3: SVM (Support Vector Machine)
print("\n--- SUPPORT VECTOR MACHINE (SVM) ---")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svm):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_svm):.4f}")

# COMPARACIÓN DE MODELOS
print("\n9. COMPARACIÓN DE MODELOS")
resultados = {
    'Decision Tree': accuracy_score(y_test, y_pred_dt),
    'Random Forest': accuracy_score(y_test, y_pred_rf),
    'SVM': accuracy_score(y_test, y_pred_svm)
}
mejor_modelo = max(resultados, key=resultados.get)
print(f"Mejor modelo: {mejor_modelo} con {resultados[mejor_modelo]:.4f} de accuracy")

print("\n✓ Análisis completado")