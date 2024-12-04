import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import toml

# Cargar las configuraciones de Supabase (si es necesario para obtener los datos)
try:
    secrets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'secrets.toml'))
    secrets = toml.load(secrets_path)
except FileNotFoundError:
    raise FileNotFoundError('El archivo secrets.toml no se encontró.')

SUPABASE_URL = secrets['SUPABASE']['URL']
SUPABASE_KEY = secrets['SUPABASE']['KEY']

# Función para cargar los datos de Supabase (también podrías cargarlo de un CSV local)
def cargar_datos_de_supabase():
    from supabase import create_client
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Cargar datos desde la tabla de Supabase
    response = supabase_client.table('datatrend_sales').select('*').execute()

    if response.error:
        raise Exception(f"Error al cargar los datos: {response.error}")
    
    # Convertir los datos a un DataFrame de pandas
    return pd.DataFrame(response.data)

# Función para entrenar el modelo de predicción
def entrenar_modelo():
    # Cargar los datos de Supabase
    df = cargar_datos_de_supabase()

    # Seleccionar las columnas relevantes para las predicciones
    X = df[['advertising', 'discount', 'season']]  # Variables independientes
    y = df['sales']  # Variable dependiente

    # Crear un conjunto de entrenamiento y un conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo de árbol de decisión
    modelo = DecisionTreeClassifier(random_state=42)

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = modelo.predict(X_test)

    # Mostrar los resultados de la evaluación
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    print("Precisión del modelo:", accuracy_score(y_test, y_pred))

    # Guardar el modelo entrenado en un archivo (por ejemplo, en formato .pkl)
    joblib.dump(modelo, 'modelo_arbol_decision.pkl')
    print("Modelo guardado exitosamente.")

# Ejecutar el entrenamiento del modelo
if __name__ == "__main__":
    entrenar_modelo()



