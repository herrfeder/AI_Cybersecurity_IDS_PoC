#!/bin/bash

aws cloudformation deploy --stack-name $1 \
--template-file $2 \
--capabilities "CAPABILITY_IAM" "CAPABILITY_NAMED_IAM" \
--parameter-overrides "WorkflowID=$3" \
--region=us-west-2
