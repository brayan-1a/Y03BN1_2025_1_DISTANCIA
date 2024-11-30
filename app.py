import streamlit as st
import pandas as pd
import datetime
from db_operations import cargar_datos, insertar_resultado_prediccion

st.title("Modelo Predictivo para proceso minero")

st.write("Datos iniciales para entrenamiento:")
datos = cargar_datos()
st.dataframe(datos)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
   
columnas_numericas = ['numero_trabajadores', 'produccion_obtenida', 'consumo_energia', 'calidad_mineral']
datos_normalizados = pd.DataFrame(scaler.fit_transform(datos[columnas_numericas]), columns=columnas_numericas)
datos.update(datos_normalizados)

st.write("Datos normalizados")
st.dataframe(datos)

st.subheader("Predicción del proceso")
if st.button("Predecir Éxito"):
    exito_predicho = True if datos['calidad_mineral'].mean() > 7 else False
    insertar_resultado_prediccion(exito_predicho)
    st.write(f"Resultado de predicción: {'Exitoso' if exito_predicho else 'No Exitoso'}")
