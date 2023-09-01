#FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

# Build time argument
ARG manifest
LABEL "manifest"=$manifest

# 8.6: meiple server gpu version 
# 7.5 7.0: aws gpu version 
ENV TORCH_CUDA_ARCH_LIST="8.6 7.5 7.0" 

RUN apt update 
RUN apt list --upgradable
RUN apt install -y wget
RUN apt autoclean
RUN apt clean 
RUN rm -rf /var/lib/apt/lists/*

# Get weight file 
COPY weights /weights
WORKDIR /weights
RUN /weights/get_weights.sh

# copy src
COPY src /app
WORKDIR /app

# Install the Python requirements.
RUN pip install -r requirements.txt

# Invoke run.py.
CMD ["python", "run.py"]