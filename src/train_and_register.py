import os
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sqlalchemy import create_engine, text

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dotenv import load_dotenv

load_dotenv()



def build_engine_from_env():
    db_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "superstore")
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD", "postgres")


    url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(url, pool_pre_ping=True)


def load_processed_superstore(engine, table_name="processed_superstore"):
    query = text(f"SELECT * FROM {table_name}")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def build_preprocess(df, target_col, explicit_num_cols=None, explicit_cat_cols=None):
    if explicit_num_cols is None or explicit_cat_cols is None:
        candidate_cols = [c for c in df.columns if c != target_col]
        num_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in candidate_cols if not pd.api.types.is_numeric_dtype(df[c])]
    else:
        num_cols = list(explicit_num_cols)
        cat_cols = list(explicit_cat_cols)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocess, num_cols, cat_cols


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    experiment_name = os.getenv("EXPERIMENT_NAME", "superstore-profit-margin")
    registered_model_name = os.getenv("REGISTERED_MODEL_NAME", "rf_profit_margin")
    model_table = os.getenv("PROCESSED_TABLE", "processed_superstore")
    target_col = os.getenv("TARGET_COL", "profit_margin")

    fast_mode = os.getenv("FAST_MODE", "0").strip().lower() in {"1", "true", "yes", "y"}

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    engine = build_engine_from_env()
    df = load_processed_superstore(engine, table_name=model_table)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in table '{model_table}'")

    df = df.dropna(subset=[target_col]).copy()

    preprocess, num_cols, cat_cols = build_preprocess(df, target_col=target_col)

    X = df[num_cols + cat_cols]
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_base = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_dist = {
        "model__n_estimators": [300, 600, 900, 1200],
        "model__max_depth": [None, 10, 20, 30, 40],
        "model__min_samples_split": [2, 5, 10, 20, 50],
        "model__min_samples_leaf": [1, 2, 4, 8, 16],
        "model__max_features": ["sqrt", 0.5, 0.8, 1.0],
        "model__bootstrap": [True, False],
    }

    n_iter = 5 if fast_mode else 30
    cv = 3 if fast_mode else 5

    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run(run_name="train_rf_random_search"):
        rf_search.fit(X_train, y_train)

        best_rf = rf_search.best_estimator_
        best_params = rf_search.best_params_
        best_cv_rmse = float(-rf_search.best_score_)

        y_pred = best_rf.predict(X_test)
        test_metrics = evaluate(y_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_rmse", best_cv_rmse)
        mlflow.log_metric("test_mae", float(test_metrics["mae"]))
        mlflow.log_metric("test_rmse", float(test_metrics["rmse"]))
        mlflow.log_metric("test_r2", float(test_metrics["r2"]))

        mlflow.sklearn.log_model(
            sk_model=best_rf,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        print("Logged training run and registered model.")
        print(f"Experiment: {experiment_name}")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Registered model: {registered_model_name}")
        print(f"Best CV RMSE: {best_cv_rmse}")
        print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
