# -*- coding: utf-8 -*-
import os
import json
import time
import logging
from typing import Any, Optional

import pika
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)

# --------------------
# Env
# --------------------
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "superstore")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "predict_jobs")

MODEL_NAME = os.getenv("MODEL_NAME", "rf_profit_margin")
TARGET_NAME = os.getenv("TARGET_NAME", "profit_margin")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/rf_tuned.joblib")
MODEL_URI = os.getenv("MODEL_URI")  # e.g. "models:/rf_profit_margin_api/3"
MODEL_VERSION = os.getenv("MODEL_VERSION", "local-joblib")

model: Any = None


# --------------------
# DB helpers
# --------------------
def get_db_conn():
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


def insert_prediction(
    raw_id: Optional[int],
    prediction: float,
) -> int:
    sql = """
        INSERT INTO predictions (raw_id, model_name, model_version, target, prediction)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (raw_id, MODEL_NAME, MODEL_VERSION, TARGET_NAME, prediction))
            row = cur.fetchone()
            conn.commit()

    if not row or "id" not in row:
        raise RuntimeError("Insert succeeded but RETURNING id failed.")
    return int(row["id"])


# --------------------
# Model loading
# --------------------
def load_model():
    global model, MODEL_VERSION

    if MODEL_URI:
        try:
            import mlflow
            import mlflow.pyfunc

            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if not mlflow_tracking_uri:
                raise RuntimeError("MLFLOW_TRACKING_URI environment variable is required when using MODEL_URI")

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            model = mlflow.pyfunc.load_model(MODEL_URI)

            if MODEL_URI.startswith("models:/"):
                MODEL_VERSION = MODEL_URI.split("/")[-1]

            logging.info(f"Loaded model from MLflow: {MODEL_URI}")
            return
        except Exception as e:
            logging.warning(f"Failed to load model from MLflow ({MODEL_URI}): {e}")
            logging.info(f"Falling back to joblib: {MODEL_PATH}")

    # Fallback to joblib
    import joblib

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Joblib model not found at {MODEL_PATH} and MLflow load failed/disabled.")

    model = joblib.load(MODEL_PATH)
    MODEL_VERSION = os.getenv("MODEL_VERSION", MODEL_VERSION)
    logging.info(f"Loaded model from joblib: {MODEL_PATH}")


def predict_one(payload: dict) -> float:
    if model is None:
        raise RuntimeError("Model is not loaded.")

    df = pd.DataFrame([payload])
    y = model.predict(df)

    # normalize to float
    try:
        return float(list(y)[0])
    except Exception:
        return float(y[0])


# --------------------
# RabbitMQ
# --------------------
def connect_with_retry():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        virtual_host=RABBITMQ_VHOST,
        credentials=credentials,
        heartbeat=30,
        blocked_connection_timeout=30,
    )

    while True:
        try:
            return pika.BlockingConnection(params)
        except Exception as e:
            logging.warning(f"RabbitMQ not ready / auth failed: {e}. Retrying in 2s...")
            time.sleep(2)


def main():
    load_model()

    connection = connect_with_retry()
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body.decode("utf-8"))

            y_pred = predict_one(payload)
            prediction_id = insert_prediction(raw_id=None, prediction=y_pred)

            logging.info(f"Job done. prediction_id={prediction_id}, y_pred={y_pred}")
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logging.exception(f"Job failed: {e}")
            # simple behavior: ack so it doesn't loop forever
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=callback, auto_ack=False)
    logging.info("Worker started. Waiting for messages...")
    channel.start_consuming()


if __name__ == "__main__":
    main()
