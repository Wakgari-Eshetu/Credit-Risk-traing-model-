# Credit-Risk-traing-model-
credit risk traing model for the bank called rati bank 

#Credit Scoring Business Understanding
Basel II and the Need for Interpretability

*The Basel II Accord emphasizes accurate risk measurement, transparency, and regulatory accountability in credit risk models. This requires models that are not only predictive but also interpretable and well-documented, allowing financial institutions and regulators to understand why a customer is classified as high or low risk. 
An interpretable model supports auditability, stress testing, and governance requirements, ensuring that risk estimates used for capital allocation and pricing can be justified and explained.

#Proxy Variable for Default and Associated Risks

*In the absence of a direct “default” label, a proxy target variable is created to approximate default-like behavior (e.g., high claim frequency or extreme loss ratios). 
This is necessary to enable supervised learning and risk estimation. However, relying on a proxy introduces business risks, including label noise, misclassification of customers, and potential bias if the proxy does not fully represent true default behavior. Poorly defined proxies can lead to incorrect pricing, unfair customer treatment, and suboptimal capital allocation.

#Model Complexity vs Interpretability Trade-offs

*Simple models such as Logistic Regression with Weight of Evidence (WoE) offer high interpretability, stability, and ease of regulatory approval, making them well-suited for regulated financial environments. However, they may sacrifice predictive power. In contrast, complex models like Gradient Boosting can capture non-linear relationships and improve accuracy but reduce transparency and are harder to explain, validate, and govern. 
In a regulated context, the trade-off involves balancing predictive performance with model explainability, compliance, and trust, often favoring simpler models for core decision-making while using complex models as supplementary tools.

# Credit Risk Probability Model

**End-to-End Machine Learning, API Deployment & CI/CD**

---

## 📌 Project Overview

This project builds an **end-to-end Credit Risk Probability Model**, starting from data exploration and preprocessing, through model training and evaluation, and finally deploying the model as a **FastAPI service** with **Docker** and **CI/CD using GitHub Actions**.

The goal is to predict the **probability of credit default** and expose the model via a REST API with automated quality checks.

---

## 🧱 Project Structure

```
credit-risk-model/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI application
│   │
│   ├── __init__.py
│   ├── predict.py               # Prediction logic
│   ├── data_processing.py       # Feature engineering & preprocessing
│   │
│   └── model/
│       └── trained_model.pkl    # Trained ML model
│
├── tests/
│   └── test_predict.py          # Unit tests
│
├── data/
│   ├── raw/                     # Raw datasets (ignored in git)
│   └── processed/               # Processed datasets
│
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis
│
├── .github/workflows/
│   └── ci.yml                   # CI pipeline
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🧪 Task Breakdown

### 🔹 Task 1: Exploratory Data Analysis (EDA)

* Data inspection and cleaning
* Distribution analysis
* Missing value handling
* Target variable understanding
* Feature insights using statistics and visualization

📁 Location: `notebooks/eda.ipynb`

---

### 🔹 Task 2: Feature Engineering

* Automatic detection of numerical and categorical features
* Encoding categorical variables
* Scaling numerical features
* Target separation
* Pipeline-ready preprocessing logic

📁 Location: `src/data_processing.py`

---

### 🔹 Task 3: Model Training

* Train credit risk prediction model
* Use scikit-learn compatible algorithms
* Probability output (`predict_proba`)
* Save trained model as `.pkl`

📁 Location: `src/model/trained_model.pkl`

---

### 🔹 Task 4: Model Evaluation & Testing

* Unit tests for feature processing
* Unit tests for prediction logic
* Dummy model usage for CI-safe testing
* Fast and deterministic test execution

📁 Location: `tests/`

Run tests:

```bash
pytest
```

---

### 🔹 Task 5: API Development (FastAPI)

* REST API for prediction
* Input validation using Pydantic
* `/predict` endpoint
* JSON-based request/response
* Swagger UI support

📁 Location: `src/api/main.py`

Run locally:

```bash
uvicorn src.api.main:app --reload
```

Docs:

```
http://localhost:8000/docs
```

---

### 🔹 Task 6: Dockerization

* Containerized FastAPI application
* Lightweight Python image
* Reproducible runtime environment
* Port exposure for API access

📁 Files:

* `Dockerfile`
* `docker-compose.yml`

Run with Docker:

```bash
docker-compose up --build
```

---

### 🔹 Task 7: CI/CD (GitHub Actions)

* Automatic pipeline on push to `main`
* Code linting using **flake8**
* Unit testing using **pytest**
* Build fails if linting or tests fail

📁 Location:

```
.github/workflows/ci.yml
```

---

## ⚙️ CI Pipeline Summary

The CI pipeline performs:

1. Code checkout
2. Python environment setup
3. Dependency installation
4. Code linting (`flake8`)
5. Unit testing (`pytest`)

This ensures **code quality and reliability** before deployment.

---

## 📦 Requirements

Install dependencies locally:

```bash
pip install -r requirements.txt
```

Main libraries:

* FastAPI
* Uvicorn
* Scikit-learn
* NumPy
* Joblib
* Pytest
* Flake8

---

## 🚀 How to Run the Full Project

### 1️⃣ Run tests

```bash
pytest
```

### 2️⃣ Run API with Docker

```bash
docker-compose up --build
```

### 3️⃣ Make a prediction

```json
POST /predict
{
  "features": [1.2, 0.5, 300, 45]
}
```

---

## ✅ Key Design Principles

* Separation of concerns (API vs ML logic)
* CI-safe testing (no real artifacts in tests)
* Reproducible builds with Docker
* Automated quality checks with CI/CD

---

## 📌 Future Improvements

* Model versioning (DVC or MLflow)
* API authentication
* Cloud deployment (AWS/GCP/Azure)
* Monitoring & logging
* Model retraining pipeline

---

## 👤 Author

**Wakgari Eshetu**
Credit Risk Modeling & ML Engineering Project
