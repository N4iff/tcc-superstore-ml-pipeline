from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any

import os
import joblib
import pandas as pd

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Superstore Profit Predictor (RF tuned)", version="1.0.0")

# --------------------
# Model loading (Joblib default, MLflow optional)
# --------------------
MODEL_NAME = os.getenv("MODEL_NAME", "rf_profit_margin")
TARGET_NAME = os.getenv("TARGET_NAME", "profit_margin")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/rf_tuned.joblib")

# If you set MODEL_URI, we load from MLflow instead of joblib
MODEL_URI = os.getenv("MODEL_URI")  # e.g. "models:/rf_profit_margin/1"
MODEL_VERSION = os.getenv("MODEL_VERSION", "local-joblib")  # will override if MLflow is used

model: Any = None

def load_model():
    global model, MODEL_VERSION

    if MODEL_URI:
        try:
            import mlflow
            import mlflow.pyfunc

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
            model = mlflow.pyfunc.load_model(MODEL_URI)

            # best-effort parse version if URI is models:/name/version
            if MODEL_URI.startswith("models:/"):
                parts = MODEL_URI.split("/")
                if len(parts) >= 3:
                    MODEL_VERSION = parts[-1]
        except Exception as e:
            raise RuntimeError(f"Failed to load MLflow model from MODEL_URI={MODEL_URI}. Error: {e}")
    else:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

load_model()

# --------------------
# Request Schemas
# --------------------
class PredictRequest(BaseModel):
    raw_id: Optional[int] = None
    sales: float
    quantity: int
    discount: float
    segment: str
    region: str
    category: str
    sub_category: str
    ship_mode: str

class PredictByRawIdRequest(BaseModel):
    raw_id: int

# --------------------
# DB helpers
# --------------------
def get_db_conn():
    """
    New connection per request.
    Supports DB_* and POSTGRES_* env names.
    """
    host = os.getenv("DB_HOST") or os.getenv("POSTGRES_HOST") or "127.0.0.1"
    port = int(os.getenv("DB_PORT") or os.getenv("POSTGRES_PORT") or "5432")
    dbname = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB")
    user = os.getenv("DB_USER") or os.getenv("POSTGRES_USER")
    password = os.getenv("DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD")

    if not all([dbname, user, password]):
        raise RuntimeError("Database env vars missing: need DB_NAME/USER/PASSWORD (or POSTGRES_DB/USER/PASSWORD)")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        cursor_factory=RealDictCursor,
    )

def db_ping() -> bool:
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                _ = cur.fetchone()
        return True
    except Exception:
        return False

def insert_prediction(
    raw_id: Optional[int],
    model_name: str,
    model_version: str,
    target: str,
    prediction: float,
) -> int:
    """
    Insert prediction row and return its id (prediction_id).
    """
    sql = """
        INSERT INTO predictions (raw_id, model_name, model_version, target, prediction)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (raw_id, model_name, model_version, target, prediction))
            row = cur.fetchone()
            conn.commit()

    if not row or "id" not in row:
        raise RuntimeError("Insert succeeded but RETURNING id failed.")
    return int(row["id"])

def fetch_raw_features(raw_id: int) -> Optional[dict]:
    sql = """
        SELECT
            sales,
            quantity,
            discount,
            segment,
            region,
            category,
            sub_category,
            ship_mode
        FROM raw_superstore
        WHERE id = %s
        LIMIT 1;
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (raw_id,))
            row = cur.fetchone()
    return row

def predict_df(x: pd.DataFrame) -> float:
    """
    Works with both sklearn (joblib) and mlflow.pyfunc models.
    """
    try:
        # sklearn Pipeline: model.predict returns ndarray
        y = model.predict(x)
        return float(y[0])
    except Exception:
        # mlflow.pyfunc: expects pandas DF and returns array-like
        y = model.predict(x)
        return float(list(y)[0])

# --------------------
# API Endpoints
# --------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "app_version": app.version,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET_NAME,
        "db_connected": db_ping(),
        "model_uri": MODEL_URI,
    }

@app.post("/predict")
def predict(req: PredictRequest):
    payload = req.model_dump()
    raw_id = payload.pop("raw_id", None)

    x = pd.DataFrame([payload])
    try:
        y_pred = predict_df(x)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    try:
        prediction_id = insert_prediction(
            raw_id=raw_id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            target=TARGET_NAME,
            prediction=y_pred,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB insert failed: {e}")

    return {
        "prediction_id": prediction_id,
        "y_pred": y_pred,
        "profit_margin_percent": round(y_pred * 100, 2),
        "logged_to_db": True,
        "raw_id": raw_id,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET_NAME,
    }

@app.post("/predict/by-raw-id")
def predict_by_raw_id(req: PredictByRawIdRequest):
    try:
        row = fetch_raw_features(req.raw_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB fetch failed: {e}")

    if row is None:
        raise HTTPException(status_code=404, detail=f"raw_id={req.raw_id} not found in raw_superstore")

    x = pd.DataFrame([row])
    try:
        y_pred = predict_df(x)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    try:
        prediction_id = insert_prediction(
            raw_id=req.raw_id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            target=TARGET_NAME,
            prediction=y_pred,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB insert failed: {e}")

    return {
        "prediction_id": prediction_id,
        "raw_id": req.raw_id,
        "y_pred": y_pred,
        "profit_margin_percent": round(y_pred * 100, 2),
        "logged_to_db": True,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET_NAME,
    }
