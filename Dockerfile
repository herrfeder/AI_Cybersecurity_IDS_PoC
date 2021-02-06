FROM python:3.8-buster

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN useradd -m datascientist
USER datascientist

WORKDIR /home/datascientist/

RUN git clone --branch david_udacity_project https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC
WORKDIR /home/datascientist/AI_Cybersecurity_IDS_PoC

ENV ENVIRONMENT production

RUN echo '#!/bin/bash' > entrypoint.sh
RUN echo 'echo "# Waiting 10 seconds for Zeek to startup; sleep 10' >> entrypoint.sh
RUN echo 'gunicorn --bind 0.0.0.0:8050 app.app:server' >> entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/home/datascientist/AI_Cybersecurity_IDS_PoC/entrypoint.sh"]
