import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

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



