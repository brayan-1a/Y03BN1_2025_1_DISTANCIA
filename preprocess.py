import pandas as pd

def load_data(supabase_client, table_name):
    """Carga datos desde Supabase."""
    response = supabase_client.table(table_name).select("*").execute()
    return pd.DataFrame(response.data)

def clean_data(df):
    """Realiza limpieza básica de los datos."""
    # Ejemplo: Manejo de valores nulos
    df.fillna(df.mean(), inplace=True)
    return df

def normalize_data(df, columns):
    """Normaliza columnas específicas."""
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


