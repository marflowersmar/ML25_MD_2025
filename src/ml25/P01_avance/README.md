# Proyecto: Predicción de Compra

## Descripción
Este proyecto tiene como objetivo predecir si un cliente realizará la compra de un nuevo producto.  
El flujo del proyecto está dividido en tres pasos principales: **exploración**, **entrenamiento** y **generación del archivo CSV de predicciones**.

---

## Archivos principales

### 1. Exploración de datos
Notebook utilizado para analizar el dataset, identificar patrones y visualizar la distribución de las variables.

[Analisis_exploratorio.ipynb](./Analisis_exploratorio.ipynb)

---

### 1.2. Procesamiento de Datos
Scripts utilizados para preparar los datos antes del entrenamiento, incluyendo limpieza, codificación y balanceo (si aplica).

[data_processing.py](./data_processing.py)  
[negative_generation.py](./negative_generation.py)

---

### 3. Entrenamiento del Modelo
Script encargado de entrenar el modelo **XGBoost**, ajustar sus parámetros y guardar el modelo resultante.

[training_xgboost.py](./training_xgboost.py)


### 4. Generación de Predicciones (Resultados)
Script encargado de utilizar el modelo entrenado para generar un archivo CSV con las predicciones finales.

[inference_xgboost.py](./inference_xgboost.py)



