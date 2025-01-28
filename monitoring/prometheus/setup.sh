helm install prometheus bitnami/kube-prometheus -f values.yaml --create-namespace -n monitoring
helm install grafana bitnami/grafana -f grafana.yaml --create-namespace -n monitoring
