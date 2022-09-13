# Tensorflow code example

This repository aims to present example of code style in tensorflow 2.0.
The example is done on binary semantic segmentation dataset.

# Installation

Create workspace folder in home directory
```
mkdir ~/deeplearning_workspace

```
and clone your repository to that folder.
```
git clone https://github.com/Dtananaev/dl_frameworks_example.git
cd ~/deeplearning_workspace

```



## Install docker

1. Install docker by following the [docker](https://docs.docker.com/engine/install/ubuntu/) instruction.
2. Install [nvidia docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) support
3. Build docker file by following command:
```
bash ~/deeplearning_workspace/frameworks_example/docker/build_image.sh
```
The docker should be build successfully.

## Seting up virtual environment

1. Create container bashrc file with following command:
```
echo "source ~/deeplearning_workspace/venv/dl_env/bin/activate" > ~/deeplearning_workspace/.bashrc
```
