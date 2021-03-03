[![CircleCI](https://circleci.com/gh/herrfeder/AI_Cybersecurity_IDS_PoC.svg?style=svg)](https://app.circleci.com/pipelines/github/herrfeder/AI_Cybersecurity_IDS_PoC/86/workflows/1c2fe7ae-4c19-412a-80f4-9c6b9cf2139a)

# AI_Cybersecurity_IDS_PoC 
and 
# Davids Udacity CloudDevOps Nanodegree Capstone Project

  * Winning Solution of BWI Data Analytics Hackathon 2020
  * CloudDevOps Pipeline with Green-Blue-Deployment for Davids Udacity CloudDevOps Nanodegree Capstone Project

![bwi_hackathon_badge](https://abload.de/img/bwi_dataanalyticshack7ujy4.png)


## App Screenshots

  * (as App is running on privately-owned real Internet-connected Infrastructure IPs are blurred)

| Monitoring Dashboard | Model Performance | Anomaly Training | Application of Models |
|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|
| ![](https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC/raw/main/screenshots/analysis_dashboard.png) | ![](https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC/raw/main/screenshots/model_performance.png) | ![](https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC/raw/main/screenshots/train_anomaly.png) | ![](https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC/raw/main/screenshots/apply_model.png) |



## Concept

  * unfortunately only in german :/

![](https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC/raw/main/concept/pitch_final.png)


## Features

  * Live-updating Webapp with DataPipeline from live running Zeek-Logs
    * extensive and easily extentable Monitoring Dashboard
  * Application of Neural Net and Random Forest models trained on pretrained labelled data against live Zeek logs
  * Training of Anomaly Detection using IsolationForest can be triggered during Runtime

## Content

  * [analysis](https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC/tree/main/analysis) contains all stuff Michael did for 
    * exploring the used labelled data from [UNSW-NB15 Datasets](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
    * checking out the performance of different models (mainly Random Forest and Neural Nets)
    * train and optimize the best model approaches using [keras-tuner](https://github.com/keras-team/keras-tuner)	

  * [app](https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC/tree/main/app) contains all stuff David did for
    * creating the live-updating Datapipeline using [zeek](https://github.com/zeek) logs
      * parsing them with an tinkered version of [ParseZeekLogs](https://github.com/dgunter/ParseZeekLogs) for enabling continuously feeding the logs into the pipeline
      * and [pygtail](https://github.com/bgreenlee/pygtail) for also continuously feeding the logs into the pipeline
    * creating Webapp using [plotly](https://github.com/plotly) and [Dash](https://github.com/plotly/dash)
    * Implementing live trained Anomaly Detection using Isolation Forest from [scikit-learn](https://github.com/scikit-learn/scikit-learn)  


## Installation/Deployment (CloudDevOps Nanodegree Part)

| CircleCI Branch CI/CD Pipeline | CircleCI Main CI/CD Pipeline |
|--------------------------------------|--------------------------------------|
| ![](https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC/raw/main/screenshots/capstone_broai_branch_pipeline.png) | ![](https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC/raw/main/screenshots/capstone_broai_main_pipeline.png) |



### Local Docker-Compose Deployment


1. Clone the repository:
    ```bash
    git clone https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC.git
    ```

2. Go into Deploy Folder and `run_compose.sh` to run `file`-based or `kafka`-based Stack:
    ```
    deploy/run_compose.sh kafka
    # OR
    deploy/run_compose.sh file
    ```

  * first run will take very long because Docker Containers will be build locally and the zeek compilation and Kafka Plugin Install will take a while 

3. Go to http://127.0.0.1:8050/


### Local Kubernetes Deployment

1. You need to build the previous Compose-based stack at least once and upload the resulting Docker Container using the `upload-docker.sh` script or you relying on my public-built Container:
  * zeek_kafka https://hub.docker.com/repository/docker/herrfeder/zeek_kafka (already in k8s Configs)
  * broai https://hub.docker.com/repository/docker/herrfeder/broai (already in k8s Configs)    
    
2. You have to prepare and start minikube and run `run_kube_local.sh`:    
    ```bash
    cd deploy
    ./run_kube_local.sh file
    # OR (you can run booth as well)
    ./run_kube_local.sh file 
    ```

3. Now add local Ingress Rule to reach the broai endpoint:
    ```bash
    kubectl apply -f broai_kubernetes/ingress-local-service.yaml
    # Check now these ingress service with
    kubectl get svc
    ```

4. Now add `green.broai` and `blue.broai` with your minikube IP to your `/etc/hosts` and visit this domains. 


### AWS Kubernetes Deployment

1. You need to build the previous Compose-based stack at least once and upload the resulting Docker Container using the `upload-docker.sh` script or you relying on my public-built Container:
  * zeek_kafka https://hub.docker.com/repository/docker/herrfeder/zeek_kafka (already in k8s Configs)
  * broai https://hub.docker.com/repository/docker/herrfeder/broai (already in k8s Configs)    

2. Install `aws-cli` and deploy the Network and Cluster Requirements with the provided AWS Cloudformation Scripts:
    ```bash
    cd .circleci

    scripts/push_cloudformation_stack.sh broainetwork cloudformation/network.yaml <your individual id>
    scripts/push_cloudformation_stack.sh broaicluster cloudformation/cluster.yaml <your individual id>
    ```
 
3. Get Access Token to acess your AWS EKS Cluster with kubectl:
    ```bash
    cd deploy

    mkdir .kube
    aws eks --region us-west-2 update-kubeconfig --kubeconfig .kube/config-aws --name AWSK8SCluster
    ``` 

4. Deploy Kubernetes Manifests:
    ```bash
    ./run_kube_aws.sh
    ```
    
4. Go to http://127.0.0.1:8050/


5. Wait for finishing and check with `kubectl --kubeconfig .kube/config-aws get svc` the resulting Loadbalancer Hostnames and access them. :)


## TODO

  * replacing filebased Datapipeline by Apache Kafka feed (DONE in scope of Davids Udacity CloudDevOps Nanodegree Capstone Project)
    * faster feeding into webapp
    * more elegant data management
  * also enabling Random Forest and Neural Net training during runtime
  * feeding predicted live-data into analysis workflow for automatic re-evaluation and re-training
