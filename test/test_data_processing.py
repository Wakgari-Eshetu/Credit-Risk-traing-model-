import pandas as pd
import numpy as np
import pytest

from src.data_processing import process_data


# Sample Test Dataset

@pytest.fixture
def sample_df():
    data = {
        'CustomerID': [1, 1, 2, 2, 3],
        'TransactionAmount': [100, 200, 150, 50, 300],
        'TransactionDate': ['2025-12-01 10:00:00',
                            '2025-12-01 15:00:00',
                            '2025-12-02 12:30:00',
                            '2025-12-02 14:45:00',
                            '2025-12-03 09:00:00'],
        'Gender': ['M', 'M', 'F', 'F', 'M'],
        'PaymentMethod': ['Cash', 'Card', 'Card', 'Cash', 'Cash'],
        'FraudResult': [0, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)
    return df


# Test the pipeline output shape

def test_process_data_shape(sample_df):
    X_transformed, pipeline = process_data(sample_df, target_col='FraudResult')
    
    # Should return a numpy array
    assert isinstance(X_transformed, np.ndarray)
    
    # Number of rows should match input
    assert X_transformed.shape[0] == sample_df.shape[0]


# Test WOE IV dictionary
def test_woe_iv(sample_df):
    _, pipeline = process_data(sample_df, target_col='FraudResult')
    
    woe_step = pipeline.named_steps['woe']
    
    # IV dictionary should exist for high-cardinality features
    assert hasattr(woe_step, 'iv_dict')
    
    # IV values should be numeric
    for iv in woe_step.iv_dict.values():
        assert isinstance(iv, float)


# Test that aggregate features exist after transform

def test_aggregate_features(sample_df):
    X_transformed, pipeline = process_data(sample_df, target_col='FraudResult')
    
    # Access aggregate features from the original dataframe after transformation
    agg = pipeline.named_steps['agg'].transform(sample_df)
    
    # Check that total_amount, avg_amount, txn_count, std_amount exist
    for col in ['total_amount', 'avg_amount', 'txn_count', 'std_amount']:
        assert col in agg.columns


# Test datetime feature extraction

def test_datetime_features(sample_df):
    dt_features = pipeline.named_steps['dt_features'].transform(sample_df)
    
    for col in ['txn_hour', 'txn_day', 'txn_month', 'txn_year']:
        assert col in dt_features.columns




if __name__ == "__main__":
    pytest.main([__file__])
