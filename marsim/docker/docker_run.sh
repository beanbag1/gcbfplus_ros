docker build -t marsim_bridge:v1.0 -f Dockerfile .

xhost local:docker &&
docker run -it --rm \
	--privileged \
	--gpus all \
	--name marsim_bridge \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$HOME/.Xauthority:/root/.Xauthority" \
    -v "$PWD/..:/root/ros_ws" \
	marsim_bridge:v1.0 \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && cd /root/ros_ws && catkin_make \
    && source devel/setup.bash && exec /bin/bash"