from supabase import create_client
import toml
import os

# Configuración del cliente de Supabase
try:
    secrets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'secrets.toml'))
    secrets = toml.load(secrets_path)
except FileNotFoundError:
    print('El archivo secrets.toml no se encontró. Asegúrate de que esté presente en el directorio raíz.')
    raise

SUPABASE_URL = secrets['SUPABASE']['URL']
SUPABASE_KEY = secrets['SUPABASE']['KEY']

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Función para insertar datos de ventas en la base de datos
def insertar_datos_ventas(datos):
    response = supabase_client.table("datatrend_sales").insert(datos).execute()
    return response

# Función para obtener todos los datos de ventas desde la base de datos
def obtener_datos_ventas():
    response = supabase_client.table("datatrend_sales").select("*").execute()
    return response.data

# Función para insertar los resultados de las predicciones de ventas
def insertar_resultado_prediccion(datos):
    response = supabase_client.table("resultados_prediccion").insert(datos).execute()
    return response
