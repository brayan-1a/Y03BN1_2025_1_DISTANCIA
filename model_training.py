import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Función para cargar los datos
def cargar_datos():
    # Cargar el archivo CSV de datos
    datos = pd.read_csv('datatrend_sales.csv')
    return datos

# Función para preprocesar los datos
def preprocesar_datos():
    # Cargar los datos
    datos = cargar_datos()

    # Seleccionar características y etiquetas
    X = datos[['advertising', 'discount', 'season']]  # Características
    y = datos['sales']  # Etiqueta (ventas)

    return X, y

# Función para entrenar el modelo
def entrenar_modelo():
    # Preprocesar los datos
    X, y = preprocesar_datos()

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo de regresión (Árbol de Decisión)
    modelo = DecisionTreeRegressor(random_state=42)
    modelo.fit(X_train, y_train)

    # Evaluar el modelo
    predicciones = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, predicciones)
    rmse = mean_squared_error(y_test, predicciones, squared=False)
    r2 = r2_score(y_test, predicciones)

    print(f"Error absoluto medio (MAE): {mae}")
    print(f"Raíz del error cuadrático medio (RMSE): {rmse}")
    print(f"R2 Score: {r2}")

    # Guardar el modelo entrenado
    joblib.dump(modelo, 'modelo_arbol_decision_regresion.pkl')

# Función para cargar el modelo entrenado
def cargar_modelo():
    # Cargar el modelo previamente guardado
    return joblib.load('modelo_arbol_decision_regresion.pkl')

# Función para guardar el modelo entrenado
def guardar_modelo(modelo):
    # Guardar el modelo en un archivo .pkl
    joblib.dump(modelo, 'modelo_arbol_decision_regresion.pkl')
