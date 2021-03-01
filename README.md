# AI_Cybersecurity_IDS_PoC

  * Winning Solution of BWI Data Analytics Hackathon 2020

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




## Installation


### Docker Instructions

1. Create directory and download Dockerfile:
    ```
    wget https://raw.githubusercontent.com/herrfeder/AI_Cybersecurity_IDS_PoC/main/Dockerfile
    ```

2. Build Docker Container:
    ```
    docker build . -t broai
    ```
    
    All models are pretrained and the build process shouldn't take too long.
    
3. Run Docker Container with mounted Zeek Volume:
    ```
    docker run -p 8050:8050 -v {zeek_location}:/home/datascientist/zeek_input broai:latest
    ```
    
    > Please be aware, for now `zeek_location` has to be provided by you and is a folder which contains your running `conn.log`

4. Go to http://127.0.0.1:8050/


## CLONE -> BUILD -> RUN -> HAPPY

## TODO

  * replacing filebased Datapipeline by Apache Kafka feed
    * faster feeding into webapp
    * more elegant data management
  * also enabling Random Forest and Neural Net training during runtime
  * feeding predicted live-data into analysis workflow for automatic re-evaluation and re-training
