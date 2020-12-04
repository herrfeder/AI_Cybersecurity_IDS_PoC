# AI_Cybersecurity_IDS_PoC

![bwi_hackathon_badge](https://abload.de/img/bwi_dataanalyticshack7ujy4.png)



## Installation

An example of this web app is temporary accessible on https://dev.datahack.de/aicspoc/ (Password Protected).
Please be gentle, the resources are restricted. This app __isn't responsive__.

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

4. Go to http://127.0.0.1:8050/aicspoc/


# CLONE -> BUILD -> RUN -> HAPPY
