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
  git \
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
  xterm 

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

RUN pip install -U "jax[cuda12]"
RUN pip install flax>=0.7.2 \
  jax>=0.4.14 \
  jraph>=0.0.6.dev0 \
  jaxtyping>=0.2.21 \
  numpy>=1.25.0 \
  einops>=0.6.1 \
  matplotlib>=3.7.2 \
  opencv-python>=4.8.0.76 \
  tqdm>=4.66.1 \
  tensorflow-probability>=0.21.0 \
  optax>=0.1.7 \
  scipy>=1.11.2 \
  wandb>=0.15.8 \
  pyyaml>=6.0.1 \
  orbax_checkpoint>=0.3.5 \
  seaborn>=0.12.2 \
  equinox>=0.11.0 \
  loguru>=0.7.2 \
  attrs>=23.1.0 \
  rich>=13.5.3 \
  ipdb==0.13.13 \
  colour>=0.1.5 \
  control>=0.9.4

RUN pip install git+https://github.com/oswinso/jaxproxqp.git 

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

