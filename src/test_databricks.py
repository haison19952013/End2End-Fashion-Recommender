import mlflow

mlflow.login()
mlflow.set_tracking_uri("databricks")

mlflow.set_experiment("/check-databricks-connection")

with mlflow.start_run():
    mlflow.log_metric("foo", 1)
    mlflow.log_metric("bar", 2)