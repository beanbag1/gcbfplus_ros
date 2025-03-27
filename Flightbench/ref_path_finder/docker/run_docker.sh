docker build -t pathfinder:noetic -f Dockerfile .

xhost +local:root

docker run -it --rm \
 --privileged \
 --gpus all \
 --net=host \
 --ipc host \
 --name pathfinder \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$HOME/.Xauthority:/root/.Xauthority" \
    -v "$PWD/../src:/root/ros_ws/src/" \
    -w "/root/ros_ws" \
 pathfinder:noetic \
 bash