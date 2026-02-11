# TCC Superstore ML Pipeline

## Overview

This project implements a **production-ready, end-to-end machine learning pipeline** for predicting **profit margin** on retail transactions using the Superstore dataset.

The system covers the full ML lifecycle:

- **Data ingestion** and transformation
- **Feature engineering** and modeling
- **Experiment tracking** and model registry (MLflow)
- **API-based inference** — both **synchronous** and **queue-based asynchronous**
- **Persistent prediction storage** in PostgreSQL
- **Cloud deployment** on **GCP** with **Managed Instance Groups (MIG)**, **Load Balancer**, and **message queueing** (RabbitMQ)

The goal is **reproducibility, observability, scalability, and clean system design** — not only model performance.

---

## Architecture

### High-Level Flow

1. **Raw data** is ingested into **PostgreSQL** (`raw_superstore`).
2. Data is cleaned and transformed into **processed_superstore**.
3. A reproducible preprocessing + modeling pipeline is trained and logged to **MLflow**.
4. The registered model is served via a **FastAPI** API and/or consumed by **workers**.
5. **Synchronous** requests: API runs inference and writes to DB immediately.
6. **Asynchronous** requests: API enqueues a job to **RabbitMQ**; a **worker** consumes the queue, runs inference, and writes to DB. Client polls **`GET /jobs/{job_id}`** for status and result.
7. All services run in **Docker**; in production they are deployed on **GCP** behind a **Load Balancer** with **MIG** for the API (and optionally workers).

### Core Components

| Component        | Role |
|-----------------|------|
| **PostgreSQL**   | Raw data, processed data, predictions, async job status (`async_jobs`) |
| **MLflow**       | Experiment tracking and model registry |
| **FastAPI**      | Inference API (sync + async enqueue) |
| **RabbitMQ**     | Message queue for async prediction jobs |
| **Worker**       | Consumes queue, runs model, writes predictions and updates job status |
| **Docker Compose**| Local / single-node deployment |
| **GCP**          | Production: **Load Balancer** → **MIG** (API instances), queueing (RabbitMQ), DB (Cloud SQL or self-hosted) |

### Sync vs Async Inference

- **Sync** (`POST /predict`, `/predict/by-raw-id`, `/predict/batch/by-raw-ids`): Request blocks until the model runs and the result is stored. Best for low latency and small batches.
- **Async** (`POST /predict_async`): API returns a `job_id` immediately after enqueueing to RabbitMQ. A **worker** processes the job and updates `async_jobs`. Client uses **`GET /jobs/{job_id}`** to poll status (`queued` → `processing` → `done` or `failed`) and get `prediction_id` when done. Best for decoupling and handling traffic spikes.

---

## Repository Structure

```text
.
├── docker/
│   └── docker-compose.yml      # db, mlflow, api, rabbitmq, worker
├── loadtest/
│   ├── locust_sync.py         # Locust load test: sync /predict
│   └── locust_async.py        # Locust load test: async /predict_async
├── notebooks/
│   └── 01_eda_processed.ipynb
├── sql/
│   └── schema.sql             # raw_superstore, processed_superstore, predictions
├── src/
│   ├── app.py                 # FastAPI: health, predict, predict_async, jobs
│   ├── worker/
│   │   └── worker.py          # RabbitMQ consumer, model inference, DB writes
│   ├── ingestion/
│   │   └── load_csv_to_db.py
│   ├── processing/
│   │   └── build_processed_table.py
│   ├── train_and_register.py
│   ├── register_model.py
│   ├── log_existing_model.py
│   └── mlflow_smoke_test.py
├── requirements.txt
├── Dockerfile                 # Used by api and worker
├── .env                       # DB, MLflow, MODEL_URI, RABBITMQ_* (not committed)
├── MLFLOW_SETUP.md            # MLflow setup and troubleshooting
└── README.md
```

---

## Data Pipeline

- **Raw:** `raw_superstore` — ingested from CSV via `src/ingestion/load_csv_to_db.py`.
- **Processed:** `processed_superstore` — built by `src/processing/build_processed_table.py`; target: **`profit_margin`**.
- **Predictions:** `predictions` — stores every prediction with `model_name`, `model_version`, `target`.
- **Async jobs:** `async_jobs` — job_id, status (`queued` | `processing` | `done` | `failed`), `prediction_id`, `error`, timestamps. Required for queue-based inference.

---

## Modeling

- **Features:** numeric (`sales`, `quantity`, `discount`), categorical (`segment`, `region`, `category`, `sub_category`, `ship_mode`).
- **Preprocessing:** `ColumnTransformer` (imputation + scaling / one-hot).
- **Model:** Tuned Random Forest; trained and logged via **MLflow**; served from registry (or joblib fallback).
- **Metrics:** MAE, RMSE, R².

---

## Experiment Tracking & Model Registry

- **MLflow** tracks parameters, metrics, artifacts, and model versions.
- The API and worker load the model from **MLflow** (e.g. `models:/rf_profit_margin/1`) for full traceability.
- See **MLFLOW_SETUP.md** for local server, registration, and troubleshooting.

---

## API Service

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/health` | Health check, model info, DB connectivity |
| POST   | `/predict` | Sync: single prediction from JSON body |
| POST   | `/predict/by-raw-id` | Sync: predict for one `raw_superstore.id` |
| POST   | `/predict/batch/by-raw-ids` | Sync: batch prediction by raw IDs |
| POST   | `/predict_async` | Async: enqueue job; returns `job_id` |
| GET    | `/jobs/{job_id}` | Get async job status and result (`prediction_id` when done) |

**Interactive docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Queue-Based Async Inference

1. Client calls **`POST /predict_async`** with the same body as `/predict`.
2. API creates a row in **`async_jobs`** with status `queued`, publishes **`{ job_id, payload }`** to **RabbitMQ** queue `predict_jobs`, and returns **`{ job_id, status: "queued" }`**.
3. A **worker** (one or more) consumes the queue, loads the model (MLflow or joblib), runs inference, inserts into **`predictions`**, and updates **`async_jobs`** to `processing` then `done` (or `failed` with `error`).
4. Client polls **`GET /jobs/{job_id}`** until `status` is `done` or `failed`, and reads `prediction_id` / `error` as needed.

This allows **horizontal scaling** of workers and **decoupling** of API from heavy inference.

---

## Load Testing

Load tests use **Locust** against the running API.

- **Sync:** `loadtest/locust_sync.py` — tasks hit **`POST /predict`**.
- **Async:** `loadtest/locust_async.py` — tasks hit **`POST /predict_async`**.

Run (from project root, with API and worker running):

```bash
# Install Locust if needed: pip install locust
locust -f loadtest/locust_sync.py --host=http://localhost:8000
# Or for async:
locust -f loadtest/locust_async.py --host=http://localhost:8000
```

Then open the URL Locust prints (e.g. http://localhost:8089), set user count and spawn rate, and start the test.

---

## Deployment

### Local (Docker Compose)

All services (PostgreSQL, MLflow, RabbitMQ, API, Worker) run via Docker Compose. Use **`--env-file`** so variables like `MODEL_URI`, `POSTGRES_*`, `RABBITMQ_*` are set.

**Start:**

```bash
cd /path/to/tcc-superstore-ml-pipeline
docker compose -f docker/docker-compose.yml --env-file .env up -d
```

Or from the `docker` directory:

```bash
cd docker
docker compose --env-file ../.env up -d
```

**Stop:**

```bash
docker compose -f docker/docker-compose.yml down
# or: cd docker && docker compose down
```

**Ports (host):**

- API: **8000**
- MLflow UI: **5001**
- PostgreSQL: **5432**
- RabbitMQ AMQP: **5672**, Management UI: **15672**

Ensure **`.env`** contains at least: `POSTGRES_*`, `DB_*`, `MLFLOW_TRACKING_URI`, `MODEL_URI`, `MODEL_PATH`, and optionally `RABBITMQ_USER` / `RABBITMQ_PASSWORD` if you configure auth.

---

### GCP Production: Load Balancer + MIG + Queueing

In production on **Google Cloud Platform**:

- **Load Balancer (e.g. HTTP(S)):** Terminates TLS and forwards traffic to the backend.
- **Managed Instance Group (MIG):** Runs multiple instances of the **API** (from the same Docker image or GCE container). The load balancer distributes requests across healthy instances. Auto-scaling can be configured based on CPU, request count, or queue depth.
- **Queueing:** **RabbitMQ** (or a managed alternative such as Cloud Pub/Sub or a RabbitMQ-on-GCE setup) is used for **async** prediction jobs. **Workers** can run in the same MIG as the API, in a separate MIG, or as separate VMs/containers that consume from the queue. The API only enqueues; workers run the model and write to the DB.
- **Database:** PostgreSQL can be **Cloud SQL** or self-managed on a VM; the API and workers connect via private IP or Cloud SQL Auth Proxy.
- **MLflow:** Either the same Docker Compose stack (for small teams) or a dedicated MLflow server (e.g. on GCE or Cloud Run) with a persistent backend (e.g. Cloud SQL, GCS artifacts).

**Typical flow:**

1. User → **Load Balancer** → one of the **API** instances in the **MIG**.
2. Sync endpoints: API runs model and responds.
3. Async: API enqueues to **RabbitMQ**, returns `job_id`. **Workers** (scaled independently) consume the queue and update **PostgreSQL** and **async_jobs**.
4. Client polls **`GET /jobs/{job_id}`** (via the same Load Balancer) for status.

This gives **scalability** (MIG + workers), **availability** (LB + health checks), and **decoupling** (queue).

---

## Reproducibility

- **Dependencies:** Pinned in `requirements.txt` (including `scikit-learn==1.5.1` to match trained models).
- **Schema:** `sql/schema.sql` is applied at DB init; ensure **`async_jobs`** exists for async flow (add migration or table if not in `schema.sql`).
- **Environment:** Copy `.env.example` (if present) to `.env` and set DB, MLflow, model URI, and RabbitMQ. Never commit `.env`.
- **Model:** Register the model to MLflow (see **MLFLOW_SETUP.md** and `src/register_model.py`) so `MODEL_URI=models:/rf_profit_margin/1` works for both API and worker.

---

## Limitations & Future Work

- Single model family (Random Forest).
- No built-in auth on the API (consider IAP, API keys, or OIDC in front of the Load Balancer).
- Monitoring is basic (health checks); consider Cloud Monitoring, Prometheus, or APM for production.
- Optional: scheduled retraining, drift detection, and Project 2 (e.g. data chatbot).

---

## Author

**Naif** — End-to-End Machine Learning & Data Engineering Project
