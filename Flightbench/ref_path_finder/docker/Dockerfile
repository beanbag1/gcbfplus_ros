## Download base image
FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]


## GPU/ display support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility

## Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  wget \
  vim \
  gcc-9 \
  g++-9 \
  xterm \
  sudo \
  python3-pip \
  python3.8 \
  python3-dev \
  python3-wheel \
  python3-opengl \
  cmake \
  pkg-config \
  ccache \
  zlib1g-dev \
  libomp-dev \
  libyaml-cpp-dev \
  libhdf5-dev \
  libgtest-dev \
  liblz4-dev \
  liblog4cxx-dev \
  libeigen3-dev
  
# Set Python 3.8 as the default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Upgrade pip to the latest version
RUN python3.8 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --user

RUN mkdir -p /home/ros_ws/src