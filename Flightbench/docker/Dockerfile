## Download base image
FROM osrf/ros:noetic-desktop-full

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
  python3-catkin-tools \
  python3-tk \
  ros-noetic-ros-numpy
  # ros-noetic-pcl-ros \
  # pcl-tools \
  # python3-pcl \
  # ros-noetic-eigen-conversions \
  # libarmadillo-dev \
  # libdynamic-reconfigure-config-init-mutex-dev \
  # libpcl-conversions-dev \
  # ros-noetic-rviz \
  # python3-opencv \
  # ros-noetic-cv-bridge \
  # ros-noetic-mavros*\
  # ros-noetic-tf2-sensor-msgs \
  # ros-noetic-tf2-geometry-msgs \
  # libdw-dev \
  # python-pyinotify \
  # protobuf-compiler \
  # libprotoc-dev

# Set Python 3.8 as the default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Upgrade pip to the latest version
RUN python3.8 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --user

RUN mkdir -p /home/ros_ws/src

## Set up bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

## Set up file path
RUN echo "export FLIGHTBENCH_PATH=/root/ros_ws/src" >> ~/.bashrc