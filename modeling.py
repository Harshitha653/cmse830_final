import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def prepare_model_data(df, target_col, feature_cols, test_size=0.2, random_state=42):
    df = df.dropna(subset=[target_col] + feature_cols)
    X = df[feature_cols].values
    y = df[target_col].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_models(X_train, y_train):
    models = {}

    models["Random Forest"] = Pipeline([
        ("s", StandardScaler()),
        ("m", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    models["Gradient Boosting"] = Pipeline([
        ("s", StandardScaler()),
        ("m", GradientBoostingRegressor(random_state=42))
    ])

    for m in models.values():
        m.fit(X_train, y_train)

    return models


def evaluate_models(models, X_train, y_train, X_test, y_test, cv_folds=5):
    rows = []
    for name, model in models.items():
        cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv_folds,
                                   scoring="neg_root_mean_squared_error").mean()

        preds = model.predict(X_test)
        rows.append({
            "Model": name,
            "CV RMSE": cv_rmse,
            "Test RMSE": mean_squared_error(y_test, preds, squared=False),
            "Test MAE": mean_absolute_error(y_test, preds),
            "Test R²": r2_score(y_test, preds),
        })
    return pd.DataFrame(rows)
