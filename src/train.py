# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# ---------------------------
# 1️⃣ Load Processed Data
# ---------------------------
data_file = "data/processed/customer_features_with_target.csv"
df = pd.read_csv(data_file)

# Separate features and target
X = df.drop(columns=['CustomerId','is_high_risk'])
y = df['is_high_risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# ---------------------------
# 2️⃣ Model Training Function
# ---------------------------
def train_models(X_train, y_train):
    models = {}
    params = {}

    # Random Forest
    models['rf'] = RandomForestClassifier(random_state=42)
    params['rf'] = {
        'n_estimators':[100, 200],
        'max_depth':[5, 10, None],
        'min_samples_split':[2, 5],
        'min_samples_leaf':[1, 2]
    }

    # Gradient Boosting (XGBoost)
    models['xgb'] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    params['xgb'] = {
        'n_estimators':[100, 200],
        'max_depth':[3, 5, 7],
        'learning_rate':[0.01, 0.1, 0.2],
        'subsample':[0.7, 1.0]
    }

    best_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        grid = GridSearchCV(model, params[name], cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"{name} best params: {grid.best_params_}")
    return best_models

# ---------------------------
# 3️⃣ Evaluation Function
# ---------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    return metrics

# ---------------------------
# 4️⃣ MLflow Tracking
# ---------------------------
def track_experiments(best_models, X_test, y_test, experiment_name="CreditRisk_Models"):
    mlflow.set_experiment(experiment_name)
    for name, model in best_models.items():
        with mlflow.start_run(run_name=name):
            metrics = evaluate_model(model, X_test, y_test)
            
            # Log parameters
            mlflow.log_params(model.get_params())
            # Log metrics
            mlflow.log_metrics(metrics)
            # Log model
            mlflow.sklearn.log_model(model, artifact_path=name+"_model")
            
            print(f"{name} logged to MLflow with metrics: {metrics}")

# ---------------------------
# 5️⃣ Main Execution
# ---------------------------
def main():
    best_models = train_models(X_train, y_train)
    track_experiments(best_models, X_test, y_test)
    print("Training and MLflow tracking completed successfully.")

if __name__ == "__main__":
    main()
