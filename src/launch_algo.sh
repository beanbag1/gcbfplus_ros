#!/bin/bash

echo "Launched gcbfplus_ros"

# Compile the code
source /opt/ros/noetic/setup.bash && cd /root/ros_ws && catkin_make \
    && source devel/setup.bash

#### Algorithm Specific Portions Start ####

# Launch the algorithm
# source src/gcbfplus_ros/.venv/bin/activate
# roslaunch gcbfplus_ros gcbfplus_ros.launch &
source src/gcbfplus_ros/.venv/bin/activate
python3 src/gcbfplus_ros/src/simulate.py --path ~/ros_ws/src/gcbfplus_ros/pretrained/LinearDrone/gcbf+ &

# Wait for algorithm to be ready
# TODO: Update to something algorithm specific
sleep 3
#### Algorithm Specific Portions End  ####

# Let sim know algo is ready
rostopic pub /ready std_msgs/Bool "data: true"