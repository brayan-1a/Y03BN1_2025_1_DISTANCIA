import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Cargar los datos
csv_path = 'datatrend_sales.csv'  # Ruta del archivo CSV de ventas
datos = pd.read_csv(csv_path)

# Seleccionar características (X) y etiquetas (y)
# En este caso, las características serán las columnas 'advertising', 'discount', 'season'
# y la etiqueta será 'sales', que es lo que queremos predecir
X = datos[['advertising', 'discount', 'season']]
y = datos['sales']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Árbol de Decisión
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo.predict(X_test)  # Predicciones en el conjunto de prueba

# Métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)  # Error absoluto medio
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Raíz del error cuadrático medio
r2 = r2_score(y_test, y_pred)  # Coeficiente de determinación R²

# Imprimir los resultados de evaluación
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Guardar el modelo entrenado
joblib.dump(modelo, 'modelo_arbol_decision.pkl')
