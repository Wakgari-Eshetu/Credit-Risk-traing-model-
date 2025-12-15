import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#Pipeline Structure for dataset 
df = pd.read_csv(r"C:\Users\HP\Downloads\Telegram Desktop\CreditRisk probabilitymodel\data\raw\creditriskmodeldata.csv")

#Identify feature types
categorical_cols = []
numerical_cols = []

for col in df.columns:
    if 'id' in col.lower():
        categorical_cols.append(col)
    elif df[col].dtype == 'object':
        categorical_cols.append(col)
    elif df[col].dtype in['int64','float64'] and df[col].nunique()<20:
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

print("Categorical col: ",categorical_cols)
print("Numerical col: ",numerical_cols)

#Create Aggregate Features per customer using groupby 
agg_features = df.groupby('CustomerId')['Amount'].agg(
    Total_Transaction_Amount='sum',
    Average_Transaction_Amount='mean',
    Transaction_Count='count',
    Std_Transaction_Amount='std'
).reset_index()

agg_features['Std_Transaction_Amount'].fillna(0, inplace=True)

print(agg_features.head())
#Merge with the original df

df = df.merge(agg_features,on="CustomerId",how='left')

print(df.head())

#Extracting Time feature 

#But first let change the TransactionStartTime to datatime 

df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
#Now lets create time feature for modeling 

df['TransactionHour'] = df['TransactionStartTime'].dt.hour #for hour 
df['TransactionDay'] = df['TransactionStartTime'].dt.day #for day
df['TransactionMonth'] = df['TransactionStartTime'].dt.month #for month
df['TransactionYear'] = df['TransactionStartTime'].dt.year #for year

# Create encoder to encode categorical variable using onehotencoder method 
#encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#encoded_array = encoder.fit_transform(df[categorical_cols])
#encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
#df_final = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

#lets handle missing value using Imputation 

