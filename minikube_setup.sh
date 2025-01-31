minikube start --driver docker --gpus all --cpus 18 --memory 20000  

minikube addons enable storage-provisioner-rancher

# Enable ingress-nginx
minikube addons enable ingress

# Enable MetalLB in Minikube
minikube addons enable metallb

# Configure MetalLB with an IP range
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  namespace: metallb-system
  name: config
data:
  config: |
    address-pools:
    - name: default
      protocol: layer2
      addresses:
      - 192.168.49.100-192.168.49.200
EOF

# Update ingress-nginx-controller to type LoadBalancer
kubectl patch svc ingress-nginx-controller -n ingress-nginx -p '{"spec": {"type": "LoadBalancer"}}'

minikube addons enable dashboard

source ./mlflow/setup.sh
source ./monitoring/setup.sh
source ./data/minio/setup.sh