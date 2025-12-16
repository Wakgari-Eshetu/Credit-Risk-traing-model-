import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------
# Sample Processed Data
# ---------------------------
@pytest.fixture
def processed_data():
    data = {
        'total_transaction_amount': [100, 200, 150, 300, 400],
        'avg_transaction_amount': [50, 100, 75, 150, 400],
        'transaction_count': [2, 2, 2, 2, 1],
        'std_transaction_amount': [0, 0, 0, 0, 0],
        'transaction_hour': [8, 9, 10, 11, 12],
        'transaction_day': [1, 2, 1, 3, 1],
        'transaction_month': [12, 12, 12, 12, 12],
        'transaction_year': [2025, 2025, 2025, 2025, 2025],
        'is_high_risk': [0, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    X = df.drop(columns=['is_high_risk'])
    y = df['is_high_risk']
    return X, y

# ---------------------------
# Test 1: Train-Test Split
# ---------------------------
def test_train_test_split(processed_data):
    X, y = processed_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

# ---------------------------
# Test 2: Random Forest Training
# ---------------------------
def test_random_forest_training(processed_data):
    X, y = processed_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)  # small estimator for test
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Check predictions shape
    assert preds.shape[0] == X_test.shape[0]
    # Check all predicted values are 0 or 1
    assert set(preds).issubset({0,1})
    
    # Compute simple metric
    acc = accuracy_score(y_test, preds)
    assert 0 <= acc <= 1

# ---------------------------
# Test 3: Gradient Boosting Training
# ---------------------------
def test_gradient_boosting_training(processed_data):
    X, y = processed_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)  # small estimator for test
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Check predictions shape
    assert preds.shape[0] == X_test.shape[0]
    # Check all predicted values are 0 or 1
    assert set(preds).issubset({0,1})
    
    # Compute simple metric
    f1 = f1_score(y_test, preds)
    assert 0 <= f1 <= 1

# ---------------------------
# Test 4: Metrics Calculation
# ---------------------------
def test_metrics_calculation(processed_data):
    X, y = processed_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
    }
    
    for metric in metrics.values():
        assert 0 <= metric <= 1
