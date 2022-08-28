#!/bin/bash

HOST_WORKSPACE="/home/$(whoami)/deeplearning_workspace/"
DOCKER_WORKSPACE="/home/deeplearning_workspace"

docker run --gpus all -ti \
--mount source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind \
--mount source=/usr/include,target=/usr/include,type=bind \
--mount source=/usr/lib,target=/usr/lib,type=bind \
--mount source=$HOST_WORKSPACE,target=$DOCKER_WORKSPACE,type=bind,consistency=cached \
--workdir=$DOCKER_WORKSPACE \
--env HOME=$DOCKER_WORKSPACE \
tensorflow_example-"$USER":latest

