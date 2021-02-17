#!/bin/bash

KUBECONFIGS="broai_kubernetes"

MINIKUBE_IP=$(minikube ip)

if [[ $1 == "file" ]]; then
  KUBEPROJECT="kube_file"
elif [[ $1 == "kafka" ]]; then
  KUBEPROJECT="kube_kafka"
elif [[ $1 == "delete" ]]; then
  kubectl delete -f "$KUBECONFIGS"/.kube_temp
  rm -r "$KUBECONFIGS"/.kube_temp
  exit
fi

if [ ! -d "$KUBECONFIGS"/.kube_temp ]; then
  mkdir "$KUBECONFIGS"/.kube_temp
else
  rm -r "$KUBECONFIGS"/.kube_temp/*
fi


cp -r "$KUBECONFIGS"/"$KUBEPROJECT"/* "$KUBECONFIGS"/.kube_temp

find "$KUBECONFIGS"/.kube_temp -type f -exec sed -i -e "s/{{EXTERNAL_IP}}/$MINIKUBE_IP/g" {} \;

kubectl apply -f "$KUBECONFIGS"/.kube_temp
  


