from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
regression_model = joblib.load(MODELS / "regression.joblib")
classification_model = joblib.load(MODELS / "classification.joblib")

app = FastAPI(title="Video Game Sales ML Demo")

class RegressionInput(BaseModel):
    na_sales: float = Field(ge=0)
    eu_sales: float = Field(ge=0)
    jp_sales: float = Field(ge=0)
    other_sales: float = Field(ge=0)

class ClassificationInput(BaseModel):
    platform: str
    year: float = Field(ge=1980, le=2035)

@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")

@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    prediction = regression_model.predict([[data.na_sales, data.eu_sales, data.jp_sales, data.other_sales]])[0]
    return {"prediction": round(float(prediction), 3), "unit": "million copies"}

@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    prediction = classification_model.predict(pd.DataFrame([{"Platform": data.platform, "Year": data.year}]))[0]
    return {"prediction": str(prediction)}
