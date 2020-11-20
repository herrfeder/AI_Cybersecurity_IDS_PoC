FROM python:3.7.6-buster

RUN pip install numpy==1.17
RUN pip install scipy
RUN pip install pandas==1.0.0
RUN pip install matplotlib
RUN pip install nltk
RUN pip install statsmodels
RUN pip install sklearn
RUN pip install tensorflow
RUN pip install keras

RUN pip install dash
RUN pip install dash-renderer
RUN pip install dash-html-components
RUN pip install dash-bootstrap-components
RUN pip install dash-core-components
RUN pip install plotly==4.5.0
RUN pip install wordcloud

RUN pip install gunicorn

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
