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

`Note: docker has root access!`

## Seting up virtual environment

1. Create container bashrc file with following command:
```
echo "source ~/dl_env/bin/activate" > ~/deeplearning_workspace/.bashrc
```
2. Activate docker:
```
bash ~/deeplearning_workspace/frameworks_example/docker/run_docker.sh
```
3. Create virtualenv:
```
python3 -m venv --system-site-packages ./dl_env
```

and activate
```
source ~/dl_env/bin/activate
```
and update pip tools:
```
pip install --upgrade pip
```
4. Install pytorch:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
5. Install tensorflow 2.0:
```
pip3 install tensorflow-gpu==2.5.0
```
6. Install requirements:
```
pip install -r dl_frameworks_example/requirements.txt
```
7. Install the package:
```
pip install -e dl_frameworks_example/
```

# Dataset

In order to run training first you need to create dataset list file:
```
cd ~/dl_frameworks_example/framework_examples
python create_dataset_list.py
```
it will create `../dataset/train.datatxt`, `../dataset/val.datatxt`, `../dataset/test.datatxt` with same data for demo purposes.

