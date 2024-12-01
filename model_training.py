import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  # Cambiar a DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Importar métricas para regresión
import joblib

# Cargar los datos
csv_path = 'datatrend_sales.csv'
datos = pd.read_csv(csv_path)

# Seleccionar características y etiquetas
X = datos[['advertising', 'discount', 'season']]  # Usamos las variables predictoras
y = datos['sales']  # La variable objetivo es 'sales' (ventas)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Árbol de Decisión para regresión
modelo = DecisionTreeRegressor(random_state=42)  # Usamos DecisionTreeRegressor para regresión
modelo.fit(X_train, y_train)

# Evaluar el modelo
predicciones = modelo.predict(X_test)
mae = mean_absolute_error(y_test, predicciones)
rmse = mean_squared_error(y_test, predicciones, squared=False)
r2 = r2_score(y_test, predicciones)

print(f"Error absoluto medio (MAE): {mae}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse}")
print(f"R2 Score: {r2}")

# Guardar el modelo entrenado
joblib.dump(modelo, 'modelo_arbol_decision_regresion.pkl')

