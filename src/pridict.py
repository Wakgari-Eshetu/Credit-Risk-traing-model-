from pathlib import Path
import joblib
import numpy as np

# Resolve base directory: src/
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "trained_model.pkl"


def load_model():
    """Load trained ML model"""
    return joblib.load(MODEL_PATH)


def predict_proba(model, X):
    """
    Predict probability of positive class
    """
    proba = model.predict_proba(X)
    return proba[:, 1]
