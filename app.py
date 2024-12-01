import streamlit as st
import pandas as pd
import datetime
from model_training import entrenar_modelo, preprocesar_datos, guardar_modelo, cargar_modelo
import matplotlib.pyplot as plt

# Cargar el archivo CSV de datos de ventas
@st.cache_data
def cargar_datos():
    return pd.read_csv('datatrend_sales.csv')

# Definir la interfaz de usuario con Streamlit
st.title("Modelo Predictivo de Ventas - DataTrend")

# Cargar datos y mostrarlos
st.subheader("Datos de ventas")
datos = cargar_datos()
st.dataframe(datos)

# Preprocesar los datos
datos_procesados = preprocesar_datos(datos)

# Entrenar el modelo y mostrar las métricas
if st.button('Entrenar Modelo'):
    modelo, mae, rmse, r2, X_test, y_test, y_pred = entrenar_modelo(datos_procesados)

    # Mostrar métricas de evaluación
    st.subheader('Métricas del Modelo')
    st.write(f'Error Absoluto Medio (MAE): {mae:.2f}')
    st.write(f'Error Cuadrático Medio (RMSE): {rmse:.2f}')
    st.write(f'R2: {r2:.2f}')

    # Graficar predicciones vs reales
    st.subheader('Comparación: Predicciones vs Reales')
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
    ax.set_xlabel('Valores Reales')
    ax.set_ylabel('Predicciones')
    ax.set_title('Predicciones vs Valores Reales')
    st.pyplot(fig)

    # Guardar el modelo
    guardar_modelo(modelo)
    st.write('Modelo entrenado y guardado exitosamente.')

# Cargar el modelo guardado y hacer una predicción
if st.button('Predecir Ventas'):
    # Cargar el modelo entrenado
    modelo_cargado = cargar_modelo()

    # Entrar valores de entrada para predecir
    publicidad = st.number_input('Ingrese el valor de publicidad')
    descuento = st.number_input('Ingrese el valor de descuento')
    temporada = st.selectbox('Seleccione la temporada', ['Verano', 'Otoño', 'Invierno', 'Primavera'])

    # Convertir temporada a formato categórico (dummy variable)
    datos_entrada = pd.DataFrame({
        'advertising': [publicidad],
        'discount': [descuento],
        'season': [temporada]
    })
    datos_entrada = pd.get_dummies(datos_entrada, drop_first=True)

    # Predecir ventas
    prediccion = modelo_cargado.predict(datos_entrada)
    st.write(f'La predicción de ventas es: {prediccion[0]:.2f}')
