FROM python:3.8-buster

RUN pip install numpy
RUN pip install scipy
RUN pip install pandas==1.0.0
RUN pip install matplotlib
RUN pip install nltk
RUN pip install statsmodels==0.11.0
RUN pip install sklearn
RUN pip install tensorflow
RUN pip install keras
RUN pip install autokeras                                                                                                                                                                
RUN pip install keras-tuner 

RUN pip install dash==1.13.4
RUN pip install dash-bootstrap-components==0.9.2
RUN pip install plotly==4.5.0
RUN pip install wordcloud

RUN pip install gunicorn
RUN pip install pygtail                                                                                                                                                                  
RUN pip install ipinfo   


RUN useradd -m datascientist
USER datascientist

WORKDIR /home/datascientist/

# have to be edited
RUN git clone https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC
WORKDIR /home/datascientist/AI_Cybersecurity_IDS_PoC
ENV ENVIRONMENT production

RUN echo '#!/bin/bash' > entrypoint.sh
RUN echo 'gunicorn --bind 0.0.0.0:8050 app.app:server' >> entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/home/datascientist/AI_Cybersecurity_IDS_PoC/entrypoint.sh"]
