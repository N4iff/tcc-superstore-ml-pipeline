"""
Script to register a model in MLflow Model Registry.

This script:
1. Connects to MLflow server running in Docker (http://127.0.0.1:5001)
2. Loads a trained model from artifacts/
3. Logs the model to MLflow with a run
4. Registers the model in Model Registry

Usage:
    python src/register_model.py

Requirements:
    - MLflow server must be running (docker compose up)
    - Model file must exist in artifacts/rf_tuned.joblib
    - .env file must contain MLFLOW_TRACKING_URI=http://127.0.0.1:5001
"""

import mlflow
import mlflow.sklearn
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "superstore-profit-margin")
REGISTERED_MODEL_NAME = os.getenv("MODEL_NAME", "rf_profit_margin")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/rf_tuned.joblib")

print(f"[INFO] Connecting to MLflow at: {MLFLOW_URI}")
print(f"[INFO] Experiment: {EXPERIMENT_NAME}")
print(f"[INFO] Model name: {REGISTERED_MODEL_NAME}")
print(f"[INFO] Model path: {MODEL_PATH}")

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_URI)

# Set or create experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Verify model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        f"Please ensure the model file exists before running this script."
    )

# Load the model
print(f"[INFO] Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print(f"[INFO] Model loaded successfully. Type: {type(model)}")

# Log and register the model
print(f"[INFO] Starting MLflow run to log and register model...")
with mlflow.start_run(run_name=f"register-{REGISTERED_MODEL_NAME}"):
    # Log some metadata
    mlflow.log_param("model_name", REGISTERED_MODEL_NAME)
    mlflow.log_param("target", "profit_margin")
    mlflow.log_param("model_type", type(model).__name__)
    
    # Log the model and register it
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    
    run_id = mlflow.active_run().info.run_id
    print(f"[INFO] Run ID: {run_id}")

print(f"✅ Successfully logged and registered model: {REGISTERED_MODEL_NAME}")
print(f"\n[INFO] You can now use this model URI in your API:")
print(f"      MODEL_URI=models:/{REGISTERED_MODEL_NAME}/1")
print(f"\n[INFO] View the model in MLflow UI:")
print(f"      {MLFLOW_URI}/#/models/{REGISTERED_MODEL_NAME}")
