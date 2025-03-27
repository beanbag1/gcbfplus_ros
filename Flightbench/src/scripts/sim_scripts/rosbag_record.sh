 #!/bin/bash 

cleanup() {
    echo "Cleaning up before exiting..."
    # Perform necessary cleanup actions here
    # trap - SIGTERM && kill -- -$$
    # rosnode list | grep record* | xargs rosnode kill
    rosnode kill /bag_recorder
    echo "Cleaned up before exiting..."
    exit 0
}

# Trap SIGTERM (sent when Docker container is stopped) and SIGINT
# trap cleanup SIGTERM SIGINT SIGKILL
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# trap "rosnode kill /bag_recorder && sleep 3s" SIGTERM SIGINT
# trap "rosnode list | grep record* | xargs rosnode kill && sleep 3" SIGTERM SIGINT

# Usage: <planner> <test_case>
# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <planner> <test_case>"
    exit 1
fi

drone_name='quad_0'
planner=$1
test_case=$2

# start rosbag recording and rename the node as bag_recorder
rosbag record /$drone_name/if_started /$drone_name/if_reached /$drone_name/if_collision \
    /$drone_name/imu /$drone_name/ground_truth/odom /$drone_name/odom \
    /$drone_name/planning/pos_cmd \
    -o /root/rosbags/${planner}_test_${test_case}.bag __name:=bag_recorder &

rosbag_pid=$(pgrep -f "rosbag record")
echo "rosbag is running with PID: $rosbag_pid"

# Keep the script running, so the container stays alive
wait -n "$rosbag_pid"
wait