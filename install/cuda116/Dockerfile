FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install curl -y &&\
  apt install git -y && \
  apt install procps -y && \
  apt install libgl1-mesa-glx -y && \
  apt install libglib2.0-0 -y && \
  apt install net-tools -y

RUN mkdir /diffusion-generator
COPY . /diffusion-generator

RUN PIP_NO_CACHE_DIR=1 pip install -r /diffusion-generator/requirements.txt

WORKDIR /diffusion-generator

ENV TRANSFORMERS_CACHE=.cache/huggingface/transformers \
  XDG_CACHE_HOME=.cache \
  MATPLOTLIBRC=.cache 

ENTRYPOINT [ "python", "web.py" ]