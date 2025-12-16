import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Aggregate Transaction Features

class TransactionAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_col='CustomerId', amount_col='Amount'):
        self.customer_col = customer_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby(self.customer_col).agg(
            total_transaction_amount=pd.NamedAgg(column=self.amount_col, aggfunc='sum'),
            avg_transaction_amount=pd.NamedAgg(column=self.amount_col, aggfunc='mean'),
            transaction_count=pd.NamedAgg(column=self.amount_col, aggfunc='count'),
            std_transaction_amount=pd.NamedAgg(column=self.amount_col, aggfunc='std')
        ).reset_index()
        agg_df['std_transaction_amount'] = agg_df['std_transaction_amount'].fillna(0)
        return agg_df


# DateTime Feature Extraction

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X.drop(columns=[self.datetime_col])


# Build Preprocessing Pipeline

def build_preprocessing_pipeline(numerical_cols, categorical_cols, encoding='label', scaling='standard'):
    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler() if scaling=='standard' else MinMaxScaler())
    ])

    # Categorical pipeline
    if encoding == 'label':
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    else:  # onehot
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    return preprocessor


# Full Task 3 Pipeline

def task3_feature_pipeline(transactions_df, numerical_cols, categorical_cols, encoding='label'):
    # 1. Aggregate customer transaction features
    aggregator = TransactionAggregator()
    customer_features = aggregator.fit_transform(transactions_df)

    # 2. Extract datetime features
    dt_features = DateTimeFeatures()
    transactions_with_dt = dt_features.fit_transform(transactions_df)

    # 3. Preprocessing
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols, encoding=encoding)
    preprocessor.fit(transactions_with_dt)

    return customer_features, preprocessor



