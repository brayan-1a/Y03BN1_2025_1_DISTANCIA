import streamlit as st
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configuración de Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Función para cargar los datos desde Supabase
def cargar_datos_desde_supabase():
    # Realizar la consulta a la base de datos
    response = supabase.table("datatrend_sales").select("*").execute()

    # Verificar si hubo un error en la respuesta
    if response.get("error"):
        st.error(f"Error al cargar los datos desde Supabase: {response['error']['message']}")
        return None
    else:
        # Devolver los datos si no hubo error
        return response.get("data")

# Llamada a la función para obtener los datos
datos = cargar_datos_desde_supabase()

# Si los datos fueron cargados correctamente, mostrarlos en la interfaz de Streamlit
if datos:
    st.write("Datos cargados desde Supabase:")
    st.write(datos)
else:
    st.write("No se pudieron cargar los datos.")

