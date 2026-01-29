from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

import joblib
import pandas as pd

import os
import psycopg2
from psycopg2.extras import RealDictCursor

from dotenv import load_dotenv
load_dotenv()



app = FastAPI(title="Superstore Profit Predictor (RF tuned)", version="1.0.0")

MODEL_PATH = "artifacts/rf_tuned.joblib"
model = joblib.load(MODEL_PATH)

MODEL_NAME = "rf_tuned"
MODEL_VERSION = "1.0.0"
TARGET_NAME = "profit_margin"


class PredictRequest(BaseModel):
    # Optional: if you already have the row id in raw_superstore, pass it to log predictions
    raw_id: Optional[int] = None

    sales: float
    quantity: int
    discount: float
    segment: str
    region: str
    category: str
    sub_category: str
    ship_mode: str


# --- DB helpers ---
def get_db_conn():
    """
    Creates a new DB connection per request.
    Supports both DB_* and POSTGRES_* env var names.
    """
    host = os.getenv("DB_HOST") or os.getenv("POSTGRES_HOST") or "127.0.0.1"
    port = int(os.getenv("DB_PORT") or os.getenv("POSTGRES_PORT") or "5432")
    dbname = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB")
    user = os.getenv("DB_USER") or os.getenv("POSTGRES_USER")
    password = os.getenv("DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        cursor_factory=RealDictCursor,
    )



def db_ping() -> bool:
    """Simple DB connectivity check."""
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
) -> None:
    """
    Inserts one prediction into the predictions table.
    raw_id can be None if we don't have an ID from raw_superstore.
    """
    sql = """
        INSERT INTO predictions (raw_id, model_name, model_version, target, prediction)
        VALUES (%s, %s, %s, %s, %s)
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (raw_id, model_name, model_version, target, prediction))
        conn.commit()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app_version": app.version,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET_NAME,
        "db_connected": db_ping(),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    # Build dataframe for model input (exclude raw_id)
    payload = req.model_dump()
    raw_id = payload.pop("raw_id", None)

    x = pd.DataFrame([payload])
    y_pred = float(model.predict(x)[0])

    # Log prediction to DB (best-effort)
    insert_prediction(
        raw_id=raw_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        target=TARGET_NAME,
        prediction=y_pred,
    )

    return {
        "y_pred": y_pred,
        "profit_margin_percent": round(y_pred * 100, 2),
        "logged_to_db": True,
        "raw_id": raw_id,
        "model_version": MODEL_VERSION,
    }
