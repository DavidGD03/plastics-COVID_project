# Diseño e implementación de un modelo computacional dinámico para evaluar el impacto ambiental de los residuos plásticos durante la emergencia sanitaria debida al COVID-19

## Trained Models

- LSTM_model.h5: Modelo entrenado con el script generatePredictions.py, el cual se usará posteriormente para predecir los tres escenarios COVID (optimista, neutral y pesimista) en el script de generatePredictionsScenarios.py

## Data

Carpeta con todas las bases de datos utilizadas en los scripts.

## Notebooks

- PRES22_simulacion_paper: Notebook con el modelo de generación de residuos usado en el paper PRES22.

- SEIRDDatasets: Notebook con la generación de infectados y muertes COVID mediante el modelo SEIR-D.

- simulacion_residuos: Notebook antiguo con distintos análisis de los datos y pruebas de simulación.

- WasteEng_simulacion_paper_RNN: Notebook con el modelo RNN utilizado en el paper WasteEng junto con el modelo de generación de residuos.

- XGBoost_model: Notebook con el modelo XGBoost para la predicción de residuos biomédicos.

## Scripts

### BMW

- generatePredictions.py: Generar predicciones de residuos biomédicos en base a tres parámetros: la región (Puducherry, Goa, Manipur, Nagaland, Mizoram o AMB), el modelo (LSTM, RNN, GRU) y el window-size (7,14,28,30).

- generatePredictionsLocal.py: Es el mismo script generatePredictions.py donde se arreglan los paths para que se pueda ejecutar de manera local.

- generatePredictionsOptimized.py: Es el mismo script generatePredictions.py pero los activadores de la red neuronal son otros para que se pueda usar la aceleración por GPU en Google Colab.

- generatePredictionsScenarios.py: Generar gráfica para el paper del modelo integrado, la cual incluye las predicciones de tres escenarios COVID (optimista, neutral y pesimista) usando el modelo LSTM entrenado previamente con el script de generatePredictions.py

### COVID Cases PRES22

- generatePredictionsPRES.py: Generar predicciones de casos COVID-19 en base a datos históricos, recibiendo como parámetros el período a predecir, el modelo (LSTM, RNN, GRU), el window-size y la cantidad de capas ocultas. Este script se usó para el paper de la conferencia PRES22.
