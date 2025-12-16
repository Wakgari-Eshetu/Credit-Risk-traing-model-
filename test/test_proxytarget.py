import pytest
import pandas as pd
import numpy as np
from src.proxytarget import task4_proxy_target, calculate_rfm, cluster_customers, assign_high_risk

# ---------------------------
# Sample Data for Testing
# ---------------------------
@pytest.fixture
def sample_data():
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'Amount': [100, 200, 150, 300, 400],
        'TransactionStartTime': [
            '2025-12-01 08:00:00',
            '2025-12-02 09:00:00',
            '2025-12-01 10:00:00',
            '2025-12-03 11:00:00',
            '2025-12-01 12:00:00'
        ]
    }
    df = pd.DataFrame(data)
    return df

# ---------------------------
# Test 1: RFM Calculation
# ---------------------------
def test_calculate_rfm(sample_data):
    rfm = calculate_rfm(sample_data)
    assert 'CustomerId' in rfm.columns
    assert all(col in rfm.columns for col in ['Recency', 'Frequency', 'Monetary'])
    # Check numeric types
    for col in ['Recency', 'Frequency', 'Monetary']:
        assert np.issubdtype(rfm[col].dtype, np.number)
    # Check frequency correctness
    freq_c1 = rfm[rfm['CustomerId'] == 'C1']['Frequency'].values[0]
    assert freq_c1 == 2

# ---------------------------
# Test 2: Clustering
# ---------------------------
def test_cluster_customers(sample_data):
    rfm = calculate_rfm(sample_data)
    rfm_clustered, kmeans_model = cluster_customers(rfm, n_clusters=2)
    assert 'cluster' in rfm_clustered.columns
    assert set(rfm_clustered['cluster'].unique()).issubset(set(range(2)))

# ---------------------------
# Test 3: High-Risk Assignment
# ---------------------------
def test_assign_high_risk(sample_data):
    rfm = calculate_rfm(sample_data)
    rfm_clustered, _ = cluster_customers(rfm, n_clusters=2)
    high_risk = assign_high_risk(rfm_clustered)
    assert 'is_high_risk' in high_risk.columns
    # Ensure binary labels
    assert set(high_risk['is_high_risk'].unique()).issubset({0,1})
    # Check that all customers are included
    assert set(high_risk['CustomerId']) == set(sample_data['CustomerId'].unique())

# ---------------------------
# Test 4: Full Task 4 Function
# ---------------------------
def test_task4_proxy_target(sample_data):
    high_risk_target = task4_proxy_target(sample_data, customer_col='CustomerId', amount_col='Amount', datetime_col='TransactionStartTime')
    assert 'CustomerId' in high_risk_target.columns
    assert 'is_high_risk' in high_risk_target.columns
    # Ensure one row per customer
    assert high_risk_target.shape[0] == sample_data['CustomerId'].nunique()
    # Ensure binary values
    assert set(high_risk_target['is_high_risk'].unique()).issubset({0,1})
