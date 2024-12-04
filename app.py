import streamlit as st
import pandas as pd
from supabase import create_client
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
import toml
import os

# Configurar el cliente de Supabase
try:
    secrets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'secrets.toml'))
    secrets = toml.load(secrets_path)
except FileNotFoundError:
    st.error('El archivo secrets.toml no se encontró. Asegúrate de que esté presente en el directorio raíz.')
    st.stop()

SUPABASE_URL = secrets['SUPABASE']['URL']
SUPABASE_KEY = secrets['SUPABASE']['KEY']

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Función para cargar los datos desde Supabase
def cargar_datos_desde_supabase():
    # Obtener los datos de la tabla 'datatrend_sales'
    response = supabase_client.table("datatrend_sales").select("*").execute()
    if response.error:
        st.error(f"Error al cargar los datos desde Supabase: {response.error}")
        return pd.DataFrame()  # Retorna un dataframe vacío en caso de error
    return pd.DataFrame(response.data)

# Función para insertar resultados de predicción
def insertar_resultado_prediccion(prediccion_exito):
    data = {
        "fecha": datetime.datetime.now().isoformat(),  # Convertir datetime a una cadena en formato ISO
        "exito_predicho": prediccion_exito
    }
    supabase_client.table("resultados_prediccion").insert(data).execute()

# Definir la interfaz de usuario con Streamlit
st.title("Modelo Predictivo para Ventas - DataTrend")

# Cargar datos desde Supabase
st.write("Datos de ventas cargados desde Supabase:")
datos = cargar_datos_desde_supabase()

if datos.empty:
    st.error("No se pudieron cargar los datos.")
else:
    st.dataframe(datos)

    # Normalización de los datos (usar columnas relevantes)
    columnas_numericas = ['advertising', 'discount']
    scaler = MinMaxScaler()
    datos_normalizados = pd.DataFrame(scaler.fit_transform(datos[columnas_numericas]), columns=columnas_numericas)
    datos.update(datos_normalizados)

    st.write("Datos normalizados:")
    st.dataframe(datos)

    # Entrenar y hacer predicciones
    st.subheader("Predicción de ventas")
    if st.button("Predecir Éxito"):
        # Usamos un modelo de árbol de decisión (puedes usar el modelo entrenado si ya lo tienes)
        try:
            # Cargar el modelo entrenado
            modelo = joblib.load('modelo_arbol_decision.pkl')
            
            # Usamos las columnas 'advertising' y 'discount' para predecir las ventas
            X = datos[['advertising', 'discount']]
            predicciones = modelo.predict(X)

            # Hacer la predicción de éxito basado en las ventas
            exito_predicho = (predicciones > 0).astype(int)  # Por ejemplo, si la predicción es positiva, consideramos éxito

            # Insertar resultados de predicción en la base de datos
            insertar_resultado_prediccion(exito_predicho)

            st.write(f"Resultado de la predicción: {'Exitoso' if exito_predicho else 'No Exitoso'}")
        except Exception as e:
            st.error(f"Error al predecir: {str(e)}")
