#!/bin/bash

echo "Launched simulator"

# Compile the code
source /opt/ros/noetic/setup.bash && cd /root/ros_ws && catkin_make \
    && source devel/setup.bash

# Usage: <planner> <test_case> <init_x> <init_y> <init_z> <init_yaw> <map> <goal_x> <goal_y> <goal_z>
# Check if the correct number of arguments is provided
if [ "$#" -lt 10 ]; then
    echo "Usage: $0 <planner> <test_case>  <init_x> <init_y> <init_z> <init_yaw> <map> <goal_x> <goal_y> <goal_z> <waypoints>"
    exit 1
fi

# Assign arguments to variables
planner=$1
test_case=$2
init_x=$3
init_y=$4
init_z=$5
init_yaw=$6
map=$7
goal_x=$8
goal_y=$9
goal_z=${10}
waypoints=${@:11}

if [ -z "$waypoints"]; then
    waypoints="0"
fi

# Wait for algorithm to be ready
topic_name="/ready"

# Polling loop until /ready topic is available
while ! rostopic info ${topic_name} > /dev/null 2>&1; do
    echo "Waiting for topic ${topic_name} to be published..."
    sleep 0.5  # Wait for 0.5 second before checking again
done

echo "Algorithm is ready!"

#### Simulation Specific Portions Start ####
# Launch simulator
roslaunch test_interface single_drone_test.launch init_x:=$init_x init_y:=$init_y init_z:=$init_z init_yaw:=$init_yaw map:=$map &

#### Simulation Specific Portions End  ####

# Wait for map to be processed by algorithm
sleep 5s

# exit_fn() {
#     trap - SIGTERM && kill -- -$$
#     sleep 3s
#     echo "Exiting Sim..."
#     exit 0
# }

# # Catch the terminate signals from outside
# trap exit_fn SIGTERM SIGINT SIGKILL 

# Start rosbag recording
exec /root/eval_scripts/rosbag_record.sh $planner $test_case &

# Tracks the odometry of the drone and publishes the next waypoints
python3 /root/eval_scripts/waypoint_tracker.py --start_x $init_x --start_y $init_y --start_z $init_z \
                                               --goal_x $goal_x --goal_y $goal_y --goal_z $goal_z \
                                               --drone_name /quad_0 \
                                               --waypoints $waypoints &
# Kill sub processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

wait

sleep 3s

# Publish goal point
# rostopic pub /goal geometry_msgs/PoseStamped "header:
#   seq: 0
#   stamp:
#     secs: 0
#     nsecs: 0
#   frame_id: 'world'
# pose:
#   position:
#     x: $goal_x
#     y: $goal_y
#     z: $goal_z
#   orientation:
#     x: 0.0
#     y: 0.0
#     z: 0.0
#     w: 0.0"