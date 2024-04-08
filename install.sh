#!/bin/bash

CUDA=cu122
#CUDA=cu102
#CUDA=cu111

conda create -n z2p python=3.8
conda activate z2p

pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
