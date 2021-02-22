#!/bin/bash

compose_folder="deploy/broai_compose"

if [[ $1 == "file" ]]; then
  if [[ $2 == "build" ]]; then
    cd $compose_folder
    docker-compose -f docker-compose.file.yml build
  fi
  
  if [[ ! -d "$compose_folder"/zeeklogs ]]
    then
      mkdir $compose_folder/zeeklogs
    else
      rm -r $compose_folder/zeeklogs
      mkdir $compose_folder/zeeklogs      
  fi	
  cd $compose_folder
  docker-compose -f docker-compose.file.yml up


elif [[ $1 == "kafka" ]]; then
  if [[ $2 == "build" ]]; then
    cd $compose_folder
    docker-compose -f docker-compose.kafka.yml build
    cd ..
    exit 0
  fi   
  
  if [[ ! -d "$compose_folder"/kafkafolder ]]
    then
      mkdir $compose_folder/kafkafolder
    else
      rm -r $compose_folder/kafkafolder
      mkdir $compose_folder/kafkafolder
    fi


  export MY_IP_ADDRESS="$(hostname -I | awk '{print $1}')"
  echo "This is the IP Address, Kafka Container will advertise: $MY_IP_ADDRESS" 
  cd $compose_folder
  docker-compose -f docker-compose.kafka.yml down
  docker-compose -f docker-compose.kafka.yml up

else
  echo -e "Please give \033[0;32mfile\033[0m or \033[0;31mkafka\033[0m as command line argument"
  exit 1
fi
