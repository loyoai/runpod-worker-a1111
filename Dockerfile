FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      fonts-dejavu-core \
      rsync \
      git \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      bc \
      libgoogle-perftools4 \
      libtcmalloc-minimal4 \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Set Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install Worker dependencies
RUN pip install requests runpod huggingface_hub

# Clone Automatic1111 repository
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /stable-diffusion-webui

# Install A1111 dependencies
WORKDIR /stable-diffusion-webui
RUN pip install -r requirements.txt

# Install IPAdapter extension
RUN git clone https://github.com/tencent-ailab/IP-Adapter.git extensions/ip-adapter

# Copy and run the cache script
COPY cache.py .
RUN python cache.py --use-cpu=all

# Add RunPod Handler and Docker container start script
COPY start.sh rp_handler.py /
COPY schemas /schemas

# Start the container
RUN chmod +x /start.sh

ENTRYPOINT /start.sh
