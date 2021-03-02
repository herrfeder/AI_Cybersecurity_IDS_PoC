# AI_Cybersecurity_IDS_PoC and Davids Udacity CloudDevOps Nanodegree Capstone Project (see DevOps Deployments on installation part)

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


### Local Docker-Compose Deployment

1. Clone the repository:
    ```bash
    git clone https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC.git
    ```

2. Go into Deploy Folder and `run_compose.sh` to run `file`-based or `kafka`-based Stack:
    ```
    cd deploy
    ./run_compose.sh kafka
    # OR
    ./run_compose.sh file
    ```

  * first run will take very long because Docker Containers will be build locally and the zeek compilation and Kafka Plugin Install will take a while 

3. Go to http://127.0.0.1:8050/


### Local Kubernetes Deployment

1. You need to build the previous Compose-based stack at least once and upload the resulting Docker Container using the `upload-docker.sh` script or you relying on my public-built Container:
  * zeek_kafka https://hub.docker.com/repository/docker/herrfeder/zeek_kafka (already in k8s Configs)
  * broai https://hub.docker.com/repository/docker/herrfeder/broai (already in k8s Configs)    
    
2. You have to prepare and start minikube and run `run_kube_local.sh`:    
    ```bash
    
    ```



## TODO

  * replacing filebased Datapipeline by Apache Kafka feed (DONE in scope of Davids Udacity CloudDevOps Nanodegree Capstone Project)
    * faster feeding into webapp
    * more elegant data management
  * also enabling Random Forest and Neural Net training during runtime
  * feeding predicted live-data into analysis workflow for automatic re-evaluation and re-training
