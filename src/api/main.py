import sys
import os

# Add the src folder to the Python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
import pandas as pd

# Import your functions from data_processing.py
from data_processing import run_task4_pipeline, process_data, build_pipeline

app = FastAPI(title="Credit Risk Full Processing API")

@app.post("/process-data")
def process_data_endpoint():
   
    # Load raw data
    df = pd.read_csv("../data/raw/creditriskmodeldata.csv")

    # Run Task 4 pipeline (RFM + clustering + is_high_risk)
    processed_df = run_task4_pipeline(df)

    # Save processed data for training
    processed_df.to_csv("data/processed/credit_risk_processed.csv", index=False)

    #  Return summary
    summary = {
        "num_rows": processed_df.shape[0],
        "num_columns": processed_df.shape[1],
        "high_risk_count": int(processed_df['is_high_risk'].sum()),
        "high_risk_percentage": round(processed_df['is_high_risk'].mean() * 100, 2)
    }

    return summary


if __name__ == "__main__":
    df = pd.read_csv("data/raw/creditriskmodeldata.csv")
    processed_df = run_task4_pipeline(df)
    processed_df.to_csv("data/processed/credit_risk_processed.csv", index=False)
    print(processed_df[['CustomerId', 'is_high_risk']].head())
