#!/bin/bash

compose_folder="broai_compose"

if [[ $1 == "file" ]]; then
  if [[ ! -d "$compose_folder"/zeeklogs ]]
    then
      mkdir $compose_folder/zeeklogs	
    fi	
  cd $compose_folder
  docker-compose -f docker-compose.file.yml up
fi
