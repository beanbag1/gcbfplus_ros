docker build -t gcbfplus_ros:noetic -f Dockerfile .

xhost +local:root

docker run -it --rm \
 --privileged \
 --gpus all \
 --net=host \
 --ipc host \
 --name gcbfplus_ros \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$HOME/.Xauthority:/root/.Xauthority" \
    -v "$PWD/..:/root/ros_ws/" \
    -w "/root/ros_ws" \
 gcbfplus_ros:noetic \
 /bin/bash -c "source /opt/ros/noetic/setup.bash && cd /root/ros_ws && catkin_make \
    && source devel/setup.bash && exec /bin/bash"