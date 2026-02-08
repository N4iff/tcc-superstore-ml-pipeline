# TCC Superstore ML Pipeline

## Overview

This project implements a **production-ready, end-to-end machine learning pipeline** for predicting **profit margin** on retail transactions using the Superstore dataset.

The system covers the full ML lifecycle:

- Data ingestion and transformation  
- Feature engineering and modeling  
- Experiment tracking and model registry  
- API-based inference  
- Persistent prediction storage  
- Cloud deployment using Docker  

The goal of the project is **not only model performance**, but also **reproducibility, observability, and clean system design**.

---

## Architecture

### High-Level Flow

1. Raw data is ingested into **PostgreSQL**
2. Data is cleaned and transformed into a processed table
3. A reproducible preprocessing + modeling pipeline is trained
4. Experiments and models are tracked and versioned in **MLflow**
5. The best model is served via a **FastAPI** service
6. Predictions are logged back into the database
7. All services are containerized and deployed on a cloud VM

### Core Components

- **PostgreSQL** – raw data, processed data, predictions  
- **scikit-learn** – preprocessing and modeling  
- **MLflow** – experiment tracking and model registry  
- **FastAPI** – inference service  
- **Docker / Docker Compose** – reproducible environment  
- **Coolify (Traefik)** – orchestration and routing  
- **GCP VM** – cloud hosting  

---

## Repository Structure

```text
.
├── docker/
│   └── docker-compose.yml
├── notebooks/
│   └── 01_eda_processed.ipynb
├── sql/
│   └── schema.sql
├── src/
│   ├── ingestion/
│   │   └── load_csv_to_db.py
│   ├── processing/
│   │   └── build_processed_table.py
│   ├── train_and_register.py
│   └── app.py
├── requirements.txt
└── README.md
```

---

## Data Pipeline

### Raw Data

- Stored in `raw_superstore`
- Contains uncleaned transactional records

### Processed Data

- Stored in `processed_superstore`
- Explicit feature selection
- Cleaned categorical and numeric columns
- Target variable: **`profit_margin`**

This separation ensures:

- **Traceability**
- **Reproducibility**
- **Safe reprocessing**

---

## Modeling

### Features

**Numeric**
- `sales`
- `quantity`
- `discount`

**Categorical**
- `segment`
- `region`
- `category`
- `sub_category`
- `ship_mode`

### Preprocessing Pipeline

- Median imputation + scaling for numeric features
- Mode imputation + one-hot encoding for categorical features
- Unified `ColumnTransformer`

### Models

- Baseline **Random Forest**
- Tuned **Random Forest** using `RandomizedSearchCV`

### Evaluation Metrics

- MAE  
- RMSE  
- R²  

All experiments and metrics are logged to **MLflow**.

---

## Experiment Tracking & Model Registry

MLflow is used to track:

- Parameters  
- Metrics  
- Artifacts  
- Model versions  

Models are registered and versioned, and inference services load models **directly from the registry**.

This ensures **full traceability** between predictions and model versions.

---

## API Service

A **FastAPI** service exposes the following endpoints:

- `POST /predict`
- `POST /predict/by-raw-id`
- `POST /predict/batch/by-raw-ids`
- `GET /health`

Each prediction:

- Runs inference using the registered model  
- Logs the result into PostgreSQL  
- Records model name and model version  

Swagger UI is available at:

```text
/docs
```

---

## Deployment

The system is deployed using **Docker** and **Docker Compose**, and hosted on a **cloud virtual machine**.

### Deployed Services

- **API** – FastAPI inference service  
- **PostgreSQL** – raw data, processed data, and predictions  
- **MLflow Tracking Server** – experiment tracking and model registry  

### Networking & Routing

- Routing and TLS termination are handled via **Coolify** and **Traefik**
- The application is exposed through a **custom domain**

---

## Reproducibility

To run the project locally:

```bash
docker compose up --build
```

All dependencies, services, and configurations are fully containerized, ensuring consistent behavior across environments.

---

## Limitations & Future Work

### Current Limitations

- Single model family (**Random Forest**)
- No automated retraining pipeline
- No authentication or authorization layer on the API
- Monitoring limited to basic health checks

### Planned Extensions

- Scheduled retraining pipelines
- Model drift detection and monitoring
- AI-powered data chatbot (**Project 2**)

---

## Author

**Naif**  
End-to-End Machine Learning & Data Engineering Project
