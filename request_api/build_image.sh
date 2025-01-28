USER=kshmawj111
IMAGE_NM=k8s_app
TAG=0.1

DOCKER_BUILDKIT=1 docker build -t $USER/$IMAGE_NM:$TAG .

docker login

docker push $USER/$IMAGE_NM:$TAG