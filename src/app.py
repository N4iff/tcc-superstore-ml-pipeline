from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Superstore Profit Predictor (RF tuned)")

MODEL_PATH = "artifacts/rf_tuned.joblib"
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    sales: float
    quantity: float
    discount: float
    segment: str
    region: str
    category: str
    sub_category: str
    ship_mode: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    x = pd.DataFrame([req.model_dump()])  # pydantic v2
    y_pred = float(model.predict(x)[0])
    return {"y_pred": y_pred}
