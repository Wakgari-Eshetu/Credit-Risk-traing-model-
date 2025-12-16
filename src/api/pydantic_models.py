from pydantic import BaseModel
from typing import List


# Request model

class PredictionRequest(BaseModel):
    features: List[float]   # processed features matching training order


# Response model

class PredictionResponse(BaseModel):
    risk_probability: float
