#!/usr/bin/env bash

if [ -z "$CIRCLECI_DOCKERHUB_USER" ];
	read -rp 'Please enter your DockerHub username: ' USERNAME
else
	USERNAME="$CIRCLECI_DOCKERHUB_USER"
fi

if [ -z "$CIRCLECI_DOCKERHUB_PASS" ];
	read -rsp 'Please enter your DockerHub password: ' PASSWORD
else
	PASSWORD="$CIRCLECI_DOCKERHUB_PASS"
fi

# Create dockerpaths
dockerpath_broai=$USERNAME/broai:latest
dockerpath_zeek=$USERNAME/zeek_kafka:latest


docker login -u $USERNAME -p $PASSWORD
docker tag broai_compose_broai:latest "$dockerpath_broai"
docker tag broai_compose_zeek:latest "$dockerpath_zeek"

docker push "$dockerpath_broai"
docker push "$dockerpath_zeek"
