SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo $SCRIPT_DIR

helm install tempo bitnami/grafana-tempo -f $SCRIPT_DIR/values.yaml --create-namespace -n monitoring