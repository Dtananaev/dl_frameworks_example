# Start from nvidia docker
FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04

# Install Python stack
RUN apt-get update \
    && apt-get --yes --no-install-recommends install \
    python3 python3-dev \
    python3-pip python3-venv python3-wheel python3-setuptools \
    build-essential cmake \
    graphviz git openssh-client \
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install zlib and jpg lib for Pillow and opencv
RUN  apt-get update && apt-get install -y zlib1g-dev libjpeg8-dev libgl1
