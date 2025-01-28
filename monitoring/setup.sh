source ./loki/setup.sh
source ./prometheus/setup.sh
source ./tempo/setup.sh

kubectl apply -f ingress.yaml -n monitoring