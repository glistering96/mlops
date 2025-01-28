SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo $SCRIPT_DIR

helm install prometheus bitnami/kube-prometheus -f $SCRIPT_DIR/values.yaml --create-namespace -n monitoring
helm install grafana bitnami/grafana -f $SCRIPT_DIR/grafana.yaml --create-namespace -n monitoring
