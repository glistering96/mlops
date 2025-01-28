SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo $SCRIPT_DIR

helm install mlflow bitnami/mlflow -f $SCRIPT_DIR/values.yaml --create-namespace -n mlflow
kubectl apply -f $SCRIPT_DIR/ingress.yaml -n mlflow

HOST=$(kubectl get ingress mlflow-ingress -n mlflow -o jsonpath='{.spec.rules[*].host}')
# 환경 변수 설정
declare -A env_vars
env_vars=(
  ["MLFLOW_TRACKING_URI"]="http://$HOST/"
  ["MLFLOW_TRACKING_USERNAME"]=$(kubectl get secret --namespace mlflow mlflow-tracking -o jsonpath="{ .data.admin-user }" | base64 -d)
  ["MLFLOW_TRACKING_PASSWORD"]=$(kubectl get secret --namespace mlflow mlflow-tracking -o jsonpath="{.data.admin-password }" | base64 -d)
)

FILE="$HOME/.bashrc"

# 각 환경 변수 추가 또는 업데이트
for VAR_NAME in "${!env_vars[@]}"; do
  VAR_VALUE="${env_vars[$VAR_NAME]}"
  # 이미 있으면 덮어쓰기, 없으면 추가
  if grep -q "^export $VAR_NAME=" "$FILE"; then
    sed -i "s|^export $VAR_NAME=.*|export $VAR_NAME=\"$VAR_VALUE\"|" "$FILE"
  else
    echo "export $VAR_NAME=\"$VAR_VALUE\"" >> "$FILE"
  fi
done

# 변경 사항 적용
source "$FILE"
