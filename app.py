import streamlit as st
from config import get_supabase_client
from preprocess import load_data, clean_data, normalize_data
from model_train import train_model, optimize_model, train_model_decision_tree
import joblib
import os

# Conexión con Supabase
supabase = get_supabase_client()

# Título
st.title("Predicción de Ventas con Streamlit y Supabase")

# Cargar datos desde Supabase
table_name = "datatrend_sales"
df = load_data(supabase, table_name)
st.write("Datos Crudos:", df)

# Verificar y procesar datos
required_columns = ["advertising", "discount", "sales"]
if all(col in df.columns for col in required_columns):
    df_clean = clean_data(df)
    df_norm = normalize_data(df_clean, ["advertising", "discount"])
    st.write("Datos Limpios y Normalizados:", df_norm)
else:
    st.error(f"Faltan columnas requeridas: {required_columns}")

# Entrenamiento del modelo
if st.button("Entrenar Modelo"):
    # Entrenar el modelo y mostrar métricas
    metrics = train_model(df_norm, target_col="sales", feature_cols=["advertising", "discount"])
    st.write("Métricas del Modelo:", metrics)

    # Realizar validación cruzada
    mean_cv_score, std_cv_score = optimize_model(df_norm, target_col="sales", feature_cols=["advertising", "discount"])
    st.write(f"Validación Cruzada - Media del MSE: {mean_cv_score}")
    st.write(f"Validación Cruzada - Desviación estándar del MSE: {std_cv_score}")

    # Confirmar si el modelo fue guardado
    if os.path.exists("model.pkl"):
        st.success("Modelo guardado exitosamente.")
    else:
        st.error("No se pudo guardar el modelo. Verifica permisos o rutas.")

# Entrenamiento del Árbol de Decisión 
if st.button("Entrenar Árbol de Decisión"):
    metrics_tree = train_model_decision_tree(df_norm, target_col="sales", feature_cols=["advertising", "discount"])
    st.write("Métricas del Árbol de Decisión:", metrics_tree)

    # Realizar validación cruzada para el Árbol de Decisión
    mean_cv_score_tree, std_cv_score_tree = optimize_model(df_norm, target_col="sales", feature_cols=["advertising", "discount"])
    st.write(f"Validación Cruzada del Árbol de Decisión - Media del MSE: {mean_cv_score_tree}")
    st.write(f"Validación Cruzada del Árbol de Decisión - Desviación estándar del MSE: {std_cv_score_tree}")

# Cargar modelo entrenado
st.write("Cargando modelo entrenado...")
model_path = "model.pkl"  # Ruta relativa del modelo
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("Modelo cargado exitosamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
else:
    st.error("El modelo no existe. Por favor, entrena el modelo primero.")




