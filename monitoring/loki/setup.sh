SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo $SCRIPT_DIR
helm install lok bitnami/grafana-loki -f $SCRIPT_DIR/values.yaml --create-namespace -n monitoring