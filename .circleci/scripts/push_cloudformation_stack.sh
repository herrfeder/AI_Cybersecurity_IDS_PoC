#!/bin/bash

STACK_EXISTS=$(aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE --query "StackSummaries[*].StackName" | grep "$1")

if [ -z "$STACK_EXISTS" ]
  then

	aws cloudformation deploy --stack-name $1 \
	--template-file $2 \
	--capabilities "CAPABILITY_IAM" "CAPABILITY_NAMED_IAM" \
	--parameter-overrides "WorkflowID=$3" \
	--region us-west-2

  else
	echo "Stack $1 already exists"
  fi


