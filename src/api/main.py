import pandas as pd
import sys
import os

# Add the src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing import task3_feature_pipeline
from proxytarget import task4_proxy_target

def main():
    # Load raw transaction data
    input_file = "data/raw/creditriskmodeldata.csv"
    df = pd.read_csv(input_file)

    # Column setup
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove('Amount')  # Amount is aggregated separately

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols.remove('CustomerId')
    categorical_cols.remove('TransactionStartTime')

    # Choose encoder: 'label' or 'onehot'
    encoding = 'label'

    # Run Task 3 pipeline
    customer_features, preprocessor = task3_feature_pipeline(df, numerical_cols, categorical_cols, encoding=encoding)

    # Save aggregated features
    output_file = "data/processed/task3_customer_features.csv"
    customer_features.to_csv(output_file, index=False)
    print(f"Task 3 customer features saved to {output_file}")
    
    high_risk_target = task4_proxy_target(df)
    
    # Merge target into features
    final_df = customer_features.merge(high_risk_target, on='CustomerId', how='left')
    final_df.to_csv("data/processed/customer_features_with_target.csv", index=False)
    print("Final customer features with proxy target saved successfully.")

if __name__ == "__main__":
    main()
