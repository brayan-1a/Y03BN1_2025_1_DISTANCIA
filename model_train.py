from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_model(df, target_col, feature_cols, save_path="model.pkl"):
    """Entrena un modelo de regresión lineal."""
    X = df[feature_cols]
    y = df[target_col]

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred),
    }

    # Guardar modelo
    joblib.dump(model, save_path)

    return metrics




