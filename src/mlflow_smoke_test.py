import mlflow
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use MLFLOW_TRACKING_URI from .env, fallback to correct port
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("superstore-profit-margin")

with mlflow.start_run(run_name="smoke-test"):
    mlflow.log_param("note", "first mlflow run")
    mlflow.log_metric("rmse", 0.123)
    print("Logged a test run to MLflow")
