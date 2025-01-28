SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo $SCRIPT_DIR

helm install minio bitnami/minio -f $SCRIPT_DIR/values.yaml --create-namespace -n data
kubectl apply -f $SCRIPT_DIR/ingress.yaml -n data