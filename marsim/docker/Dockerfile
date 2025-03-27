# docker build -t marsim_bridge:v1.0 -f Dockerfile .

# ros2 foxy, ros1 noetic and bridge
FROM ros:foxy-ros1-bridge

SHELL ["/bin/bash", "-c"]

## GPU/ display support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
vim \
tmux \
wget \
unzip \
git \
net-tools \
libglfw3-dev \
libglew-dev \
python3-pip \
libarmadillo-dev \
libdw-dev \
python3-opencv \
libopencv-dev \
ros-noetic-cmake-modules \
ros-noetic-pcl-ros \
pcl-tools \
ros-noetic-cv-bridge \
ros-noetic-rviz 

# nlopt for fuel
# RUN git clone https://github.com/stevengj/nlopt.git && cd nlopt && mkdir build && cd build && cmake .. && make && sudo make install

RUN mkdir -p /root/ros_ws/src
WORKDIR /root/ros_ws

# ADD MARSIM /root/ros_ws/src/MARSIM/
# RUN cd src/MARSIM/fuel_planner/bspline_opt && wget -O CMakeLists.txt https://raw.githubusercontent.com/engcang/exploration-algorithms/main/fuel/CMakeLists.txt

# Source ROS1 setup for catkin_make
RUN source /opt/ros/noetic/setup.bash && catkin_init_workspace src

# Build the ROS workspace using catkin_make
RUN source /opt/ros/noetic/setup.bash && catkin_make


RUN echo "source ~/ros_ws/devel/setup.bash" >> ~/.bashrc
RUN echo "alias ros2='source /opt/ros/foxy/setup.bash'" >> ~/.bashrc