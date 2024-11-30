import pandas as pd
import supabase
import datetime
from supabase import create_client
import toml
import os
import streamlit as st

# Configurar el cliente de Supabase
try:
    secrets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'secrets.toml'))
    secrets = toml.load(secrets_path)
except FileNotFoundError:
    st.error('El archivo secrets.toml no se encontró. Asegúrate de que esté presente en el directorio raíz.')
    st.stop()

SUPABASE_URL = secrets['SUPABASE_URL']
SUPABASE_KEY = secrets['SUPABASE_KEY']

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data
def cargar_datos():
    return pd.read_csv('mining_data.csv')

def insertar_resultado_prediccion(prediccion_exito):
    data = {
        "fecha": datetime.datetime.now(),
        "exito_predicho": prediccion_exito
    }
    supabase_client.table("resultados_prediccion").insert(data).execute()
