#!/bin/bash

HOST_WORKSPACE="/home/$(whoami)/deeplearning_workspace/"
DOCKER_WORKSPACE="/home/deeplearning_workspace"

docker run -it \
--runtime nvidia \
--mount source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind \
--mount source=$HOST_WORKSPACE,target=$DOCKER_WORKSPACE,type=bind,consistency=cached \
--workdir=$DOCKER_WORKSPACE \
--env HOME=$DOCKER_WORKSPACE \
dl_frameworks_example-"$USER":latest

