# Primero, realiza las importaciones al principio del archivo (fuera de las funciones)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Función para entrenar el modelo de regresión lineal
def train_model(df, target_col, feature_cols, save_path="model.pkl"):
    # Separar datos
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred),
    }

    # Guardar el modelo
    joblib.dump(model, save_path)

    # Confirmar si se guardó
    if os.path.exists(save_path):
        print(f"Modelo guardado exitosamente en: {save_path}")
    else:
        print("Error: No se pudo guardar el modelo.")

    return metrics

# Función para realizar validación cruzada
def optimize_model(df, target_col, feature_cols):
    """Implementación de la validación cruzada."""
    X = df[feature_cols]
    y = df[target_col]
    
    model = DecisionTreeRegressor(random_state=42, max_depth=5)  # Usamos el modelo de árbol de decisión con límite en la profundidad

    # Realizamos la validación cruzada con 5 particiones
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    
    # Convertimos los valores negativos de MSE a positivos
    cv_scores = -cv_scores  # Negamos el MSE para obtener valores positivos
    mean_cv_score = cv_scores.mean()
    std_cv_score = cv_scores.std()

    return mean_cv_score, std_cv_score

# Función para entrenar el modelo de árbol de decisión
def train_model_decision_tree(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    
    # Entrenamiento de Árbol de Decisión con límite en la profundidad
    model_tree = DecisionTreeRegressor(random_state=42, max_depth=5)
    model_tree.fit(X, y)

    # Evaluar el modelo
    y_pred_tree = model_tree.predict(X)
    metrics_tree = {
        "MAE": mean_absolute_error(y, y_pred_tree),
        "RMSE": mean_squared_error(y, y_pred_tree, squared=False),
        "R2": r2_score(y, y_pred_tree),
    }

    # Guardar modelo (opcional)
    joblib.dump(model_tree, "model_tree.pkl")

    return metrics_tree






