
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo $SCRIPT_DIR

source $SCRIPT_DIR/prometheus/setup.sh
source $SCRIPT_DIR/loki/setup.sh
source $SCRIPT_DIR/tempo/setup.sh

kubectl apply -f $SCRIPT_DIR/ingress.yaml -n monitoring