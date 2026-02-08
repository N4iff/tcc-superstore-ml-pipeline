FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY sql/ ./sql

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

ARG BUILD_SHA=dev
RUN echo "BUILD_SHA=$BUILD_SHA"


CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
