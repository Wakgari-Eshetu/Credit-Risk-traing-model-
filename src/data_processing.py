import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

from xverse.transformer import WOE  

#Transaction Aggregator
class TransactionAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerID', amount_col='TransactionAmount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        agg = X.groupby(self.customer_id_col)[self.amount_col].agg(
            total_amount='sum',
            avg_amount='mean',
            txn_count='count',
            std_amount='std'
        ).reset_index()
        X = X.merge(agg, on=self.customer_id_col, how='left')
        return X

#Datetime Features

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionDate'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        X['txn_hour'] = X[self.datetime_col].dt.hour
        X['txn_day'] = X[self.datetime_col].dt.day
        X['txn_month'] = X[self.datetime_col].dt.month
        X['txn_year'] = X[self.datetime_col].dt.year
        return X

#WOE Encoder for High-Cardinality Features

class WOEEncoderXverse(BaseEstimator, TransformerMixin):
    def __init__(self, woe_cols, target):
        self.woe_cols = woe_cols
        self.target = target
        self.woe_encoder = None

    def fit(self, X, y=None):
        self.woe_encoder = WOE(columns=self.woe_cols)
        self.woe_encoder.fit(X, X[self.target])
        return self

    def transform(self, X):
        return self.woe_encoder.transform(X)

#Build the Full Pipeline

def build_pipeline(low_card_cat_cols, numerical_cols, woe_cols, target_col):
    # Pipeline for low-cardinality categorical features
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Pipeline for numerical features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, low_card_cat_cols),
        ('num', num_pipeline, numerical_cols)
    ])

    # Full pipeline
    full_pipeline = Pipeline([
        ('agg', TransactionAggregator()),
        ('dt_features', DateTimeFeatures()),
        ('woe', WOEEncoderXverse(woe_cols=woe_cols, target=target_col)),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline

#Optional: Fit & Transform Function

def process_data(df, low_card_cat_cols, numerical_cols, woe_cols, target_col):
    pipeline = build_pipeline(low_card_cat_cols, numerical_cols, woe_cols, target_col)
    X_transformed = pipeline.fit_transform(df)
    return X_transformed

import pandas as pd


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby('CustomerId')
        .agg(
            Recency=('TransactionStartTime',
                     lambda x: (snapshot_date - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Amount', 'sum')
        )
        .reset_index()
    )

    return rfm


def create_proxy_target(rfm: pd.DataFrame) -> pd.DataFrame:
    
    rfm = rfm.copy()

    # Percentile thresholds
    recency_thr = rfm['Recency'].quantile(0.80)
    freq_thr = rfm['Frequency'].quantile(0.20)
    monetary_thr = rfm['Monetary'].quantile(0.20)

    rfm['high_recency'] = (rfm['Recency'] >= recency_thr).astype(int)
    rfm['low_frequency'] = (rfm['Frequency'] <= freq_thr).astype(int)
    rfm['low_monetary'] = (rfm['Monetary'] <= monetary_thr).astype(int)

    rfm['credit_risk_target'] = (
        rfm[['high_recency', 'low_frequency', 'low_monetary']]
        .sum(axis=1) >= 2
    ).astype(int)

    return rfm[['CustomerId', 'credit_risk_target']]


def merge_target(df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    
    return df.merge(target_df, on='CustomerId', how='left')

def cluster_customers_rfm(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
    ) -> pd.DataFrame:
   

    rfm = rfm.copy()

    # Select RFM features
    rfm_features = rfm[['Recency', 'Frequency', 'Monetary']]

    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    # K-Means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    return rfm
def profile_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    
    return (
        rfm
        .groupby('cluster')[['Recency', 'Frequency', 'Monetary']]
        .mean()
        .reset_index()
    )

def identify_high_risk_cluster(cluster_profile: pd.DataFrame) -> int:
   
    profile = cluster_profile.copy()

    # Normalize for fair comparison
    profile['recency_rank'] = profile['Recency'].rank(ascending=False)
    profile['frequency_rank'] = profile['Frequency'].rank(ascending=True)
    profile['monetary_rank'] = profile['Monetary'].rank(ascending=True)

    profile['risk_score'] = (
        profile['recency_rank']
        + profile['frequency_rank']
        + profile['monetary_rank']
    )

    # Cluster with highest risk score
    high_risk_cluster = profile.loc[
        profile['risk_score'].idxmax(), 'cluster'
    ]

    return int(high_risk_cluster)

def assign_high_risk_label(
    rfm: pd.DataFrame,
    high_risk_cluster: int
    ) -> pd.DataFrame:
   
    rfm = rfm.copy()
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    return rfm

def integrate_target(
    df: pd.DataFrame,
    rfm_with_target: pd.DataFrame
    ) -> pd.DataFrame:

    target_df = rfm_with_target[['CustomerId', 'is_high_risk']]

    processed_df = df.merge(
        target_df,
        on='CustomerId',
        how='left'
    )

    return processed_df

# Add this at the bottom of data_processing.py
def run_task4_pipeline(df: pd.DataFrame) -> pd.DataFrame:

    rfm = calculate_rfm(df)
    rfm = cluster_customers_rfm(rfm)
    cluster_profile = profile_clusters(rfm)
    high_risk_cluster = identify_high_risk_cluster(cluster_profile)
    rfm = assign_high_risk_label(rfm, high_risk_cluster)
    processed_df = integrate_target(df, rfm)
    return processed_df
