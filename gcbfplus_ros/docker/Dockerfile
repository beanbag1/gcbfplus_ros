FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-c"]

## GPU/ display support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
python3.10 \ 
python3.10-venv \
python3.10-distutils

RUN python3.10 -m venv ~/ros_ws/src/gcbfplus_ros/.venv
ENV PATH="~/ros_ws/src/gcbfplus_ros/.venv/bin:$PATH"
RUN echo "source ~/ros_ws/src/gcbfplus_ros/.venv/bin/activate" >> ~/.bashrc

## Install dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  wget \
  vim \
  ros-noetic-pcl-ros \
  pcl-tools \
  python3-pcl \
  ros-noetic-ros-numpy \ 
  ros-noetic-eigen-conversions \
  libarmadillo-dev \
  libdynamic-reconfigure-config-init-mutex-dev \
  libpcl-conversions-dev \
  ros-noetic-rviz \
  python3-opencv \
  ros-noetic-cv-bridge \
  ros-noetic-mavros*\
  ros-noetic-tf2-sensor-msgs \
  ros-noetic-tf2-geometry-msgs \
  libdw-dev \
  gcc-9 \
  g++-9 \
  python-pyinotify \
  python3-pip \
  protobuf-compiler \
  libprotoc-dev \
  python3-catkin-tools \
  sudo \
  xterm \
  git

# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"

# RUN arch=$(uname -m) && \
#     if [ "$arch" = "x86_64" ]; then \
#     MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
#     elif [ "$arch" = "aarch64" ]; then \
#     MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
#     else \
#     echo "Unsupported architecture: $arch"; \
#     exit 1; \
#     fi && \
#     wget $MINICONDA_URL -O miniconda.sh && \
#     mkdir -p /root/.conda && \
#     bash miniconda.sh -b -p /root/miniconda3 && \
#     rm -f miniconda.sh

# RUN conda --version
# RUN conda create -n gcbfplus_ros python=3.10
# RUN echo "conda activate env" > ~/.bashrc

# ENV PATH="/opt/conda/envs/env/bin:$PATH"
# SHELL ["/bin/bash", "--login", "-c"]

RUN pip install --no-cache-dir "jax[cuda12]"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# RUN pip install -e

RUN mkdir -p /home/$USERNAME/ros_ws/src

#testing!
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
# software-properties-common

# RUN add-apt-repository ppa:deadsnakes/ppa

# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
# python3.10

# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
#testing

## Set up bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source ~/ros_ws/devel/setup.bash" >> ~/.bashrc

