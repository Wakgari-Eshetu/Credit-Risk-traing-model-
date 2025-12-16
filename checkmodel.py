import mlflow

# List all registered models
client = mlflow.MlflowClient()
for rm in client.search_registered_models():
    print("Model name:", rm.name)
    for v in rm.latest_versions:
        print("  Version:", v.version, "Stage:", v.current_stage)

from mlflow.tracking import MlflowClient

client = MlflowClient()

registered_models = client.search_registered_models()
for rm in registered_models:
    print("Model name:", rm.name)
    for v in rm.latest_versions:
        print("  Version:", v.version, "Stage:", v.current_stage)
