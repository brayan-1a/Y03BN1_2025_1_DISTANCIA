import streamlit as st
import pandas as pd
from model_training import entrenar_modelo, preprocesar_datos, cargar_modelo
import datetime
from supabase import create_client
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

# Cargar el archivo CSV de datos iniciales
@st.cache_data
def cargar_datos():
    return pd.read_csv('datatrend_sales.csv')

# Función para insertar el resultado de la predicción en la base de datos
def insertar_resultado_prediccion(prediccion_exito):
    prediccion_exito_str = 'True' if prediccion_exito else 'False'
    
    data = {
        "fecha": datetime.datetime.now().isoformat(),  # Convertir datetime a una cadena en formato ISO
        "exito_predicho": prediccion_exito_str  # Guardar el valor como cadena ('True' o 'False')
    }
    
    # Insertar los datos en Supabase
    supabase_client.table("resultados_prediccion2").insert(data).execute()

# Definir la interfaz de usuario con Streamlit
st.title("Modelo Predictivo para Ventas de Productos Electrónicos")

# Cargar datos y mostrarlos
st.write("Datos de ventas:")
datos = cargar_datos()
st.dataframe(datos)

# Preprocesamiento de los datos
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

columnas_numericas = ['advertising', 'discount', 'season']
datos_normalizados = pd.DataFrame(scaler.fit_transform(datos[columnas_numericas]), columns=columnas_numericas)

datos.update(datos_normalizados)

st.write("Datos normalizados:")
st.dataframe(datos)

# Entrenamiento del modelo
st.subheader("Entrenamiento del Modelo Predictivo")

if st.button("Entrenar el Modelo"):
    entrenar_modelo()
    st.write("El modelo ha sido entrenado y guardado correctamente.")

# Cargar el modelo y hacer predicciones
st.subheader("Realizar Predicción de Ventas")

if st.button("Predecir Ventas"):
    modelo = cargar_modelo()
    
    # Seleccionar características de ejemplo para hacer la predicción
    datos_nuevos = pd.DataFrame({
        'advertising': [5000],  # Ejemplo de publicidad
        'discount': [15],  # Ejemplo de descuento
        'season': [2]  # Ejemplo de temporada
    })
    
    prediccion = modelo.predict(datos_nuevos)
    
    st.write(f"Predicción de ventas: {prediccion[0]:.2f} unidades")
    
    # Convertir la predicción ajustada a un valor booleano (ejemplo con umbral de 0.5)
    prediccion_exito = prediccion[0] > 0.01  # Si la predicción es mayor a 0.01, consideramos que es exitosa
    
    # Insertar el resultado de la predicción en la base de datos de Supabase
    insertar_resultado_prediccion(prediccion_exito)

