#!/bin/bash

conda create -p ./python310 python=3.10.10

conda activate python310/

pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install -U -I --no-deps xformers==0.0.17 -i https://mirrors.aliyun.com/pypi/simple/

cd points/sd-scripts
pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install lion-pytorch dadaptation -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install lycoris-lora -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install fastapi uvicorn -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install wandb -i https://mirrors.bfsu.edu.cn/pypi/web/simple

cd ../datasets/anime-segmentation
pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple

cd ../auto-tagging
pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
