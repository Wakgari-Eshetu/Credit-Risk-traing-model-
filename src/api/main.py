import numpy as np
import mlflow
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Import Pydantic models from separate file
from pydantic_models import PredictionRequest, PredictionResponse


# Lifespan for loading MLflow model

MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Production"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded from MLflow Registry")
    yield
    # Optional cleanup


# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="REST API for predicting credit risk probability",
    version="1.0",
    lifespan=lifespan
)


# Health check endpoint

@app.get("/")
def health_check():
    return {"status": "API is running"}


# Prediction endpoint

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    X = np.array(request.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    return PredictionResponse(risk_probability=float(prediction))
