import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Calculate RFM

def calculate_rfm(df, customer_col='CustomerId', amount_col='Amount', date_col='TransactionStartTime', snapshot_date=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    
    # Frequency & Monetary
    agg_df = df.groupby(customer_col).agg(
        Frequency=(amount_col, 'count'),
        Monetary=(amount_col, 'sum')
    ).reset_index()
    
    # Recency
    recency_df = df.groupby(customer_col)[date_col].max().reset_index()
    recency_df['Recency'] = (snapshot_date - recency_df[date_col]).dt.days
    recency_df = recency_df.drop(columns=[date_col])
    
    # Merge all
    rfm = agg_df.merge(recency_df, on=customer_col)
    return rfm


# 2. Scale RFM

def scale_rfm(rfm_df):
    scaler = StandardScaler()
    rfm_scaled = rfm_df.copy()
    rfm_scaled[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(
        rfm_scaled[['Recency', 'Frequency', 'Monetary']]
    )
    return rfm_scaled, scaler

# 3. Cluster Customers

def cluster_customers(rfm_scaled_df, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_scaled_df['cluster'] = kmeans.fit_predict(rfm_scaled_df[['Recency','Frequency','Monetary']])
    return rfm_scaled_df, kmeans


# 4. Assign High-Risk Label

def assign_high_risk_label(rfm_scaled_df, original_rfm_df):
    cluster_stats = original_rfm_df.groupby('cluster').agg({
        'Recency':'mean',
        'Frequency':'mean',
        'Monetary':'mean'
    }).reset_index()
    
    high_risk_cluster = cluster_stats.sort_values(
        ['Recency','Frequency','Monetary'], ascending=[False, True, True]
    ).iloc[0]['cluster']
    
    original_rfm_df['is_high_risk'] = (original_rfm_df['cluster'] == high_risk_cluster).astype(int)
    return original_rfm_df[['CustomerId','is_high_risk']]

# 5. Full Task 4 Pipeline

def task4_proxy_target(transactions_df, snapshot_date=None):
    rfm = calculate_rfm(transactions_df, snapshot_date=snapshot_date)
    rfm_scaled, _ = scale_rfm(rfm)
    rfm_scaled, _ = cluster_customers(rfm_scaled)
    
    rfm_with_cluster = rfm.copy()
    rfm_with_cluster['cluster'] = rfm_scaled['cluster']
    
    target_df = assign_high_risk_label(rfm_scaled, rfm_with_cluster)
    return target_df
