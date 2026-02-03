import mlflow
import mlflow.sklearn
import joblib
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use MLFLOW_TRACKING_URI from .env, fallback to correct port (5001 on host)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME = "superstore-profit-margin"
REGISTERED_MODEL_NAME = "rf_profit_margin"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

model = joblib.load("artifacts/rf_tuned.joblib")

with mlflow.start_run(run_name="log-existing-rf"):
    mlflow.log_param("model_name", REGISTERED_MODEL_NAME)
    mlflow.log_param("target", "profit_margin")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    print("✅ Logged & registered:", REGISTERED_MODEL_NAME)
