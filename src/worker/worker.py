import os
import json
import time
import logging

import pika
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
QUEUE_NAME = os.getenv("RABBITMQ_QUEUE", "predict_jobs")

MODEL_NAME = os.getenv("MODEL_NAME", "rf_profit_margin")
TARGET_NAME = os.getenv("TARGET_NAME", "profit_margin")
MODEL_URI = os.getenv("MODEL_URI")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/rf_tuned.joblib")

MODEL_VERSION = os.getenv("MODEL_VERSION", "local-joblib")
model = None


def get_db_conn():
    host = os.getenv("DB_HOST") or "db"
    port = int(os.getenv("DB_PORT") or "5432")
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


def insert_prediction(prediction: float) -> int:
    sql = """
        INSERT INTO predictions (raw_id, model_name, model_version, target, prediction)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (None, MODEL_NAME, MODEL_VERSION, TARGET_NAME, prediction))
            row = cur.fetchone()
            conn.commit()
    return int(row["id"])


def load_model():
    global model, MODEL_VERSION

    if MODEL_URI:
        import mlflow
        import mlflow.pyfunc

        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not mlflow_tracking_uri:
            raise RuntimeError("MLFLOW_TRACKING_URI is required when using MODEL_URI")

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        model = mlflow.pyfunc.load_model(MODEL_URI)

        if MODEL_URI.startswith("models:/"):
            MODEL_VERSION = MODEL_URI.split("/")[-1]

        logging.info(f"Loaded model from MLflow: {MODEL_URI}")
        return

    import joblib
    model = joblib.load(MODEL_PATH)
    logging.info(f"Loaded model from joblib: {MODEL_PATH}")


def predict_one(payload: dict) -> float:
    df = pd.DataFrame([payload])
    y = model.predict(df)
    return float(y[0])


def connect_with_retry():
    while True:
        try:
            return pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        except Exception as e:
            logging.warning(f"RabbitMQ not ready yet: {e}. Retrying...")
            time.sleep(2)


def main():
    load_model()

    connection = connect_with_retry()
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    def callback(ch, method, properties, body):
        try:
            payload = json.loads(body.decode("utf-8"))
            y_pred = predict_one(payload)
            prediction_id = insert_prediction(y_pred)
            logging.info(f"Job done. prediction_id={prediction_id}, y_pred={y_pred}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logging.exception(f"Job failed: {e}")
            # Ack anyway for now (simple). Later we can add retry/DLQ.
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=False)
    logging.info("Worker started. Waiting for messages...")
    channel.start_consuming()


if __name__ == "__main__":
    main()
