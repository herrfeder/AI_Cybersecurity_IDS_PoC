#!/usr/bin/env bash

read -rp 'Please enter your DockerHub username: ' USERNAME
echo ""
read -rsp 'Please enter your DockerHub password: ' PASSWORD
echo ""

# Create dockerpaths
dockerpath_broai=$USERNAME/broai:latest
dockerpath_zeek=$USERNAME/zeek_kafka:latest


docker login -u $USERNAME -p $PASSWORD
docker tag broai_compose_broai:latest "$dockerpath_broai"
docker tag broai_compose_zeek:latest "$dockerpath_zeek"

docker push "$dockerpath_broai"
docker push "$dockerpath_zeek"
