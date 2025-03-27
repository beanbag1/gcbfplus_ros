echo "Launched simulator"

# Compile the code
source /opt/ros/noetic/setup.bash && cd /root/ros_ws && catkin_make \
    && source devel/setup.bash

# Usage: <planner> <test_case> <num_drones>
# Check if the correct number of arguments is provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <planner> <test_case> <num_drones>"
    exit 1
fi

source /root/ros_ws/docker/config.sh

# Assign arguments to variables
planner=$1
test_case=$2
num_drones=$3

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
roslaunch test_interface n_drone_test.launch drone_num:=$num_drones init_x_0:=$init_x_0 init_y_0:=$init_y_0 init_z_0:=$init_z_0 init_yaw_0:=$init_yaw_0 \
                                                                    init_x_1:=$init_x_1 init_y_1:=$init_y_1 init_z_1:=$init_z_1 init_yaw_1:=$init_yaw_1 \
                                                                    init_x_2:=$init_x_2 init_y_2:=$init_y_2 init_z_2:=$init_z_2 init_yaw_1:=$init_yaw_2 \
                                                                    init_x_3:=$init_x_3 init_y_3:=$init_y_3 init_z_3:=$init_z_3 init_yaw_1:=$init_yaw_3 \
                                                                    init_x_4:=$init_x_4 init_y_4:=$init_y_4 init_z_4:=$init_z_4 init_yaw_1:=$init_yaw_4 \
                                                                    init_x_5:=$init_x_5 init_y_5:=$init_y_5 init_z_5:=$init_z_5 init_yaw_1:=$init_yaw_5 \
                                                                    init_x_6:=$init_x_6 init_y_6:=$init_y_6 init_z_6:=$init_z_6 init_yaw_1:=$init_yaw_6 \
                                                                    init_x_7:=$init_x_7 init_y_7:=$init_y_7 init_z_7:=$init_z_7 init_yaw_1:=$init_yaw_7 \
                                                                    map:=$map &

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
# exec /root/eval_scripts/rosbag_record.sh $planner $test_case &

# this is criminal
arg_arr=(
    "--start_x $init_x_0 --start_y $init_y_0 --start_z $init_z_0 \
    --goal_x $goal_x_0 --goal_y $goal_y_0 --goal_z $goal_z_0 \
    --drone_name /quad_0 \
    --waypoints $waypoints_0"
    "--start_x $init_x_1 --start_y $init_y_1 --start_z $init_z_1 \
    --goal_x $goal_x_1 --goal_y $goal_y_1 --goal_z $goal_z_1 \
    --drone_name /quad_1 \
    --waypoints $waypoints_1"
    "--start_x $init_x_2 --start_y $init_y_2 --start_z $init_z_2 \
    --goal_x $goal_x_2 --goal_y $goal_y_2 --goal_z $goal_z_2 \
    --drone_name /quad_2 \
    --waypoints $waypoints_2"
    "--start_x $init_x_3 --start_y $init_y_3 --start_z $init_z_3 \
    --goal_x $goal_x_3 --goal_y $goal_y_3 --goal_z $goal_z_3 \
    --drone_name /quad_3 \
    --waypoints $waypoints_3"
    "--start_x $init_x_4 --start_y $init_y_4 --start_z $init_z_4 \
    --goal_x $goal_x_4 --goal_y $goal_y_4 --goal_z $goal_z_4 \
    --drone_name /quad_4 \
    --waypoints $waypoints_4"
    "--start_x $init_x_5 --start_y $init_y_5 --start_z $init_z_5 \
    --goal_x $goal_x_5 --goal_y $goal_y_5 --goal_z $goal_z_5 \
    --drone_name /quad_5 \
    --waypoints $waypoints_5"
    "--start_x $init_x_6 --start_y $init_y_6 --start_z $init_z_6 \
    --goal_x $goal_x_6 --goal_y $goal_y_6 --goal_z $goal_z_6 \
    --drone_name /quad_6 \
    --waypoints $waypoints_6"
    "--start_x $init_x_7 --start_y $init_y_7 --start_z $init_z_7 \
    --goal_x $goal_x_7 --goal_y $goal_y_7 --goal_z $goal_z_7 \
    --drone_name /quad_7 \
    --waypoints $waypoints_7"
)

# Tracks the odometry of the drone and publishes the next waypoints
for i in $(seq 0 $((num_drones-1))); do
    echo "running waypoint tracker with arguments: ${arg_arr[$i]}"
    python3 /root/eval_scripts/waypoint_tracker.py ${arg_arr[$i]} &
done

# Kill sub processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

wait

sleep 3s