import streamlit as st
from config import get_supabase_client
from preprocess import load_data, clean_data, normalize_data
from model_train import train_model
import joblib

# Conexión con Supabase
supabase = get_supabase_client()

# Título
st.title("Predicción de Ventas con Streamlit y Supabase")

# Cargar y mostrar datos
table_name = "datatrend_sales"
df = load_data(supabase, table_name)
st.write("Datos Crudos:", df)

# Preprocesamiento
df_clean = clean_data(df)
df_norm = normalize_data(df_clean, ["Publicidad", "Descuentos"])
st.write("Datos Limpios y Normalizados:", df_norm)

# Entrenamiento del modelo
if st.button("Entrenar Modelo"):
    metrics = train_model(df_norm, target_col="Ventas", feature_cols=["Publicidad", "Descuentos"])
    st.write("Métricas del Modelo:", metrics)

# Cargar modelo entrenado
try:
    model = joblib.load("model.pkl")
    st.success("Modelo cargado exitosamente")
except FileNotFoundError:
    st.error("No se ha entrenado un modelo aún.")


