FROM python:3.8-buster

ENV REPO=AI_Cybersecurity_IDS_PoC
COPY requirements.txt .

RUN pip install -r requirements.txt

RUN useradd -m datascientist
USER datascientist

WORKDIR /home/datascientist/

ENV ENVIRONMENT production

RUN echo '#!/bin/bash' > entrypoint.sh
RUN echo     "rm $REPO; \
	     git clone --branch \$PROJECT_BRANCH https://github.com/herrfeder/AI_Cybersecurity_IDS_PoC $REPO; \
 	     cd $REPO; \
	     echo '# Waiting 10 seconds for Zeek to startup'; sleep 10; \
	     pwd;\
	     gunicorn --bind 0.0.0.0:8050 app.app:server" >> entrypoint.sh

RUN chmod +x entrypoint.sh
RUN cat entrypoint.sh

ENTRYPOINT ["/home/datascientist/entrypoint.sh"]
