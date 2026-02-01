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
    """
    Try MLflow if MODEL_URI is set; otherwise fallback to joblib.
    If MLflow fails for any reason, we fallback to joblib to keep the API available.
    """
    global model, MODEL_VERSION

    # 1) Try MLflow
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

            print(f"[INFO] Loaded model from MLflow: {MODEL_URI}")
            return
        except Exception as e:
            print(f"[WARN] Failed to load MLflow model from MODEL_URI={MODEL_URI}. "
                  f"Falling back to joblib. Error: {e}")

    # 2) Fallback to joblib
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    # keep MODEL_VERSION from env if provided; otherwise "local-joblib"
    MODEL_VERSION = os.getenv("MODEL_VERSION", MODEL_VERSION)
    print(f"[INFO] Loaded model from joblib: {MODEL_PATH}")


load_model()

# --------------------
# Request Schemas
# --------------------
class PredictRequest(BaseModel):
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


class PredictBatchByRawIdsRequest(BaseModel):
    raw_ids: list[int]


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


def insert_predictions_bulk(
    rows: list[tuple[int, str, str, str, float]],
) -> int:
    """
    Bulk insert predictions.
    rows tuples: (raw_id, model_name, model_version, target, prediction)
    Returns number of inserted rows.
    """
    if not rows:
        return 0

    sql = """
        INSERT INTO predictions (raw_id, model_name, model_version, target, prediction)
        VALUES (%s, %s, %s, %s, %s)
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()

    return len(rows)


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


def fetch_raw_features_batch(raw_ids: list[int]) -> list[dict]:
    """
    Fetch multiple rows from raw_superstore by ids.
    Returns list of dict rows.
    """
    if not raw_ids:
        return []

    sql = """
        SELECT
            id AS raw_id,
            sales,
            quantity,
            discount,
            segment,
            region,
            category,
            sub_category,
            ship_mode
        FROM raw_superstore
        WHERE id = ANY(%s);
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (raw_ids,))
            rows = cur.fetchall()

    return rows or []


def predict_df(x: pd.DataFrame) -> list[float]:
    """
    Works with both sklearn (joblib) and mlflow.pyfunc models.
    Returns list[float] predictions (supports batch).
    """
    y = model.predict(x)
    # normalize to python floats
    try:
        return [float(v) for v in list(y)]
    except Exception:
        # fallback
        return [float(y[0])]


def chunk_list(items: list[int], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def dedupe_preserve_order(items: list[int]) -> list[int]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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

    x = pd.DataFrame([payload])
    try:
        y_pred = predict_df(x)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    try:
        prediction_id = insert_prediction(
            raw_id=None,  # manual/ad-hoc prediction has no raw_id
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
        y_pred = predict_df(x)[0]
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


@app.post("/predict/batch/by-raw-ids")
def predict_batch_by_raw_ids(req: PredictBatchByRawIdsRequest):
    MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))

    raw_ids = req.raw_ids
    if not raw_ids:
        raise HTTPException(status_code=400, detail="raw_ids must not be empty")

    raw_ids = dedupe_preserve_order(raw_ids)

    if len(raw_ids) > MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"Too many raw_ids. Max allowed is {MAX_BATCH}. Got {len(raw_ids)}."
        )

    all_results = []
    total_logged = 0
    total_found = 0

    for chunk in chunk_list(raw_ids, CHUNK_SIZE):
        try:
            rows = fetch_raw_features_batch(chunk)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"DB fetch failed: {e}")

        if not rows:
            continue

        df = pd.DataFrame(rows)
        total_found += len(df)

        raw_id_col = df["raw_id"].tolist()
        x = df.drop(columns=["raw_id"])

        try:
            preds = predict_df(x)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

        try:
            total_logged += insert_predictions_bulk([
                (rid, MODEL_NAME, MODEL_VERSION, TARGET_NAME, pred)
                for rid, pred in zip(raw_id_col, preds)
            ])
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"DB insert failed: {e}")

        all_results.extend([
            {
                "raw_id": rid,
                "y_pred": pred,
                "profit_margin_percent": round(pred * 100, 2),
            }
            for rid, pred in zip(raw_id_col, preds)
        ])

    if not all_results:
        raise HTTPException(status_code=404, detail="No matching raw_ids found in raw_superstore")

    return {
        "count_requested": len(raw_ids),
        "count_found": total_found,
        "count_returned": len(all_results),
        "count_logged": total_logged,
        "max_batch": MAX_BATCH,
        "chunk_size": CHUNK_SIZE,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET_NAME,
        "results": all_results,
    }
