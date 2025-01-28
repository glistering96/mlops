helm install minio bitnami/minio -f values.yaml --create-namespace -n data
kubectl apply -f ingress.yaml -n data