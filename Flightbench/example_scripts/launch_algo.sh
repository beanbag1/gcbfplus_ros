#!/bin/bash

echo "Launched Fast Planner"

# Compile the code
source /opt/ros/noetic/setup.bash && cd /root/ros_ws && catkin_make \
    && source devel/setup.bash

#### Algorithm Specific Portions Start ####

# Launch the algorithm
roslaunch plan_manage kino_replan_marsim.launch &

# Wait for algorithm to be ready
# TODO: Update to something algorithm specific
sleep 3
#### Algorithm Specific Portions End  ####

# Let sim know algo is ready
rostopic pub /ready std_msgs/Bool "data: true"