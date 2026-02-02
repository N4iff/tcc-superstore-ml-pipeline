# MLflow Setup Guide

## نظرة عامة

المشروع يستخدم MLflow لتتبع التجارب وتسجيل الموديلات. MLflow يعمل داخل Docker container.

## البنية (Architecture)

```
┌─────────────────┐
│   Notebook/PC    │  mlflow.set_tracking_uri("http://127.0.0.1:5001")
│   (Host)         │
└────────┬─────────┘
         │ HTTP
         │ Port 5001
         ▼
┌─────────────────┐
│  MLflow Server  │  Inside: http://mlflow:5000
│  (Docker)        │  Host: http://127.0.0.1:5001
│  Port: 5001:5000 │
└────────┬─────────┘
         │ Internal Docker Network
         ▼
┌─────────────────┐
│   FastAPI       │  MLFLOW_TRACKING_URI=http://mlflow:5000
│   (Docker)       │  MODEL_URI=models:/rf_profit_margin/1
└─────────────────┘
```

## تشغيل MLflow Server

### الطريقة الصحيحة:

```bash
# من مجلد docker/
cd docker
docker compose --env-file ../.env up

# أو من جذر المشروع:
docker compose -f docker/docker-compose.yml --env-file .env up
```

**ملاحظة مهمة:** يجب استخدام `--env-file` حتى يتم تحميل المتغيرات مثل `${MODEL_URI}` و `${POSTGRES_DB}` في docker-compose.yml.

### التحقق من أن MLflow يعمل:

افتح المتصفح على:
```
http://127.0.0.1:5001
```

يجب أن ترى واجهة MLflow UI.

## تسجيل الموديل في MLflow

### الطريقة الموصى بها:

استخدم السكربت الجاهز:

```bash
python src/register_model.py
```

هذا السكربت:
1. يتصل بـ MLflow على `http://127.0.0.1:5001`
2. يحمل الموديل من `artifacts/rf_tuned.joblib`
3. يسجله في MLflow مع run
4. ينشئ Registered Model باسم `rf_profit_margin`
5. ينشئ Version 1

### من داخل Notebook:

```python
import mlflow
import mlflow.sklearn
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

# استخدام MLFLOW_TRACKING_URI من .env
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_experiment("superstore-profit-margin")

model = joblib.load("artifacts/rf_tuned.joblib")

with mlflow.start_run(run_name="my-run"):
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="rf_profit_margin",
    )
```

## متغيرات البيئة المهمة

في ملف `.env`:

```env
# للـ Notebook/Scripts على الجهاز:
MLFLOW_TRACKING_URI=http://127.0.0.1:5001

# للـ API داخل Docker (يتم تعيينه تلقائياً في docker-compose.yml):
# MLFLOW_TRACKING_URI=http://mlflow:5000

# URI الموديل المسجل:
MODEL_URI=models:/rf_profit_margin/1
```

## استكشاف الأخطاء

### المشكلة: "Registered Model not found"

**السبب:** الموديل لم يُسجل بعد في MLflow، أو تم تسجيله في سيرفر MLflow آخر.

**الحل:**
1. تأكد أن MLflow server يعمل: `http://127.0.0.1:5001`
2. شغّل `python src/register_model.py` لتسجيل الموديل
3. تحقق من وجود الموديل في MLflow UI: `http://127.0.0.1:5001/#/models/rf_profit_margin`

### المشكلة: "OSError: Read-only file system: '/mlflow'"

**السبب:** محاولة الكتابة مباشرة على `/mlflow/...` من خارج Docker.

**الحل:** استخدم فقط HTTP tracking URI:
```python
mlflow.set_tracking_uri("http://127.0.0.1:5001")  # ✅ صحيح
# لا تستخدم:
# mlflow.set_tracking_uri("file:/mlflow")  # ❌ خطأ
# mlflow.set_tracking_uri("sqlite:////mlflow/mlflow.db")  # ❌ خطأ
```

### المشكلة: "POSTGRES_DB not set" أو "MODEL_URI not set"

**السبب:** docker compose لم يحمل `.env` بشكل صحيح.

**الحل:** استخدم `--env-file`:
```bash
docker compose -f docker/docker-compose.yml --env-file .env up
```

### المشكلة: الـ API يفشل عند الإقلاع

**الأسباب المحتملة:**
1. `MODEL_URI` غير موجود في `.env`
2. الموديل غير مسجل في MLflow
3. MLflow server غير متاح

**الحل:**
1. تأكد من وجود `MODEL_URI=models:/rf_profit_margin/1` في `.env`
2. سجّل الموديل أولاً: `python src/register_model.py`
3. تأكد أن MLflow service يعمل قبل تشغيل API

## الملفات المهمة

- `docker/docker-compose.yml` - إعدادات Docker services
- `.env` - متغيرات البيئة
- `src/register_model.py` - سكربت لتسجيل الموديل
- `src/app.py` - FastAPI application (يستخدم MLflow لتحميل الموديل)
- `src/mlflow_smoke_test.py` - اختبار بسيط لـ MLflow
- `src/log_existing_model.py` - سكربت بديل لتسجيل الموديل

## خطوات العمل الكاملة

1. **شغّل Docker services:**
   ```bash
   cd docker
   docker compose --env-file ../.env up -d
   ```

2. **تحقق من MLflow UI:**
   ```
   http://127.0.0.1:5001
   ```

3. **سجّل الموديل:**
   ```bash
   python src/register_model.py
   ```

4. **تحقق من تسجيل الموديل:**
   - افتح MLflow UI
   - اذهب إلى Models tab
   - يجب أن ترى `rf_profit_margin` مع Version 1

5. **شغّل API:**
   ```bash
   # API سيعمل تلقائياً مع docker compose
   # أو شغّله يدوياً:
   docker compose -f docker/docker-compose.yml up api
   ```

6. **اختبر API:**
   ```bash
   curl http://localhost:8000/health
   ```

## ملاحظات مهمة

- **لا تستخدم `/mlflow/...` أبداً** من notebooks أو scripts على الجهاز
- **استخدم فقط HTTP URIs** للاتصال بـ MLflow
- **تأكد من استخدام نفس MLflow server** لتسجيل الموديل والتحميل منه
- **Port 5001 على الـ host** = **Port 5000 داخل Docker network**
