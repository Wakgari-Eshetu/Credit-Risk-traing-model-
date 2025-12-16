import pytest
import pandas as pd
import numpy as np
from src.data_processing import task3_feature_pipeline

# ---------------------------
# Sample DataFrame for testing
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
        ],
        'ProductCategory': ['A', 'B', 'A', 'B', 'C']
    }
    df = pd.DataFrame(data)
    return df

# ---------------------------
# Test Task 3 Pipeline
# ---------------------------
def test_task3_feature_pipeline(sample_data):
    numerical_cols = ['Amount']
    categorical_cols = ['ProductCategory']

    customer_features, preprocessor = task3_feature_pipeline(
        sample_data, numerical_cols, categorical_cols
    )

    # ---------------------------
    # Test 1: Aggregated features exist
    # ---------------------------
    expected_columns = [
        'CustomerId',
        'total_transaction_amount',
        'avg_transaction_amount',
        'transaction_count',
        'std_transaction_amount'
    ]
    for col in expected_columns:
        assert col in customer_features.columns, f"{col} missing in aggregated features"

    # ---------------------------
    # Test 2: Preprocessor can transform data
    # ---------------------------
    transformed = preprocessor.transform(sample_data)
    assert transformed.shape[0] == sample_data.shape[0], "Row count mismatch after preprocessing"

    # ---------------------------
    # Test 3: Label encoding converts categorical columns to numeric
    # ---------------------------
    cat_idx = categorical_cols.index('ProductCategory')
    encoded_column = transformed[:, len(numerical_cols) + cat_idx]  # position in ColumnTransformer output
    assert np.issubdtype(encoded_column.dtype, np.integer) or np.issubdtype(encoded_column.dtype, np.floating), \
        "Label encoding failed for categorical column"

    # ---------------------------
    # Test 4: DateTime features are correctly extracted
    # ---------------------------
    dt_features = ['transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year']
    # Use the DateTime transformer directly
    from src.data_processing import DateTimeFeatures
    dt_transformer = DateTimeFeatures(datetime_col='TransactionStartTime')
    dt_result = dt_transformer.transform(sample_data)
    for col in dt_features:
        assert col in dt_result.columns, f"{col} missing after DateTime feature extraction"
        assert dt_result[col].dtype in [np.int64, np.int32], f"{col} has wrong dtype"

    # ---------------------------
    # Test 5: Aggregated numeric values are correct
    # ---------------------------
    c1_features = customer_features[customer_features['CustomerId'] == 'C1']
    assert c1_features['total_transaction_amount'].values[0] == 300, "Total transaction amount incorrect"
    assert c1_features['transaction_count'].values[0] == 2, "Transaction count incorrect"
