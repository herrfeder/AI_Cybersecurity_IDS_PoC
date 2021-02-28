#!/bin/bash

KUBECONFIGS="broai_kubernetes"



kubectl --kubeconfig=.kube/config-aws apply -f "$KUBECONFIGS"/kube_aws/
  
kubectl --kubeconfig=.kube/config-aws apply -f "$KUBECONFIGS"/loadbalancer-aws-service-green-main.yaml


