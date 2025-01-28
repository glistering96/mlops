helm install mlflow bitnami/mlflow -f values.yaml --create-namespace -n mlflow
kubectl apply -f ingress.yaml -n data