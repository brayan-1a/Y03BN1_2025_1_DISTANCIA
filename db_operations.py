from supabase import create_client
import os
import toml

# Cargar los secretos desde el archivo `secrets.toml`
try:
    secrets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'secrets.toml'))
    secrets = toml.load(secrets_path)
except FileNotFoundError:
    raise FileNotFoundError('El archivo secrets.toml no se encontró.')

SUPABASE_URL = secrets['SUPABASE']['URL']
SUPABASE_KEY = secrets['SUPABASE']['KEY']

# Crear el cliente de Supabase
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Función para insertar los resultados de las predicciones en la base de datos
def insertar_resultado_prediccion(prediccion_exito):
    """
    Inserta el resultado de la predicción en la tabla 'resultados_prediccion' de Supabase.
    """
    from datetime import datetime

    # Preparamos los datos que se van a insertar
    data = {
        "fecha": datetime.now().isoformat(),  # Usamos el formato ISO para la fecha
        "exito_predicho": prediccion_exito
    }

    # Insertamos los datos en la tabla 'resultados_prediccion'
    try:
        response = supabase_client.table("resultados_prediccion").insert(data).execute()
        if response.error:
            raise Exception(f"Error al insertar en Supabase: {response.error}")
        return response.data
    except Exception as e:
        print(f"Error al insertar los resultados: {e}")
        return None

