#!/usr/bin/env python3
import rospy
import argparse
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import math

class WaypointNavigator:
    def __init__(self, start_pose, goal_pose, waypoints, drone_name):
        self.drone_name = drone_name
        self.current_pose = None
        self.start_pose = start_pose
        self.goal_pose = goal_pose
        self.waypoints = waypoints
        self.current_waypoint_index = 0

        # Publishers for distance (as Float32) and waypoints
        self.start_dist_pub = rospy.Publisher(f'{self.drone_name}/if_started', Float32, queue_size=10)
        self.end_dist_pub = rospy.Publisher(f'{self.drone_name}/if_reached', Float32, queue_size=10)
        self.waypoint_pub = rospy.Publisher(f'{self.drone_name}/goal', PoseStamped, queue_size=10)

        # Subscriber for odometry
        rospy.Subscriber(f'{self.drone_name}/odom', Odometry, self.odom_callback)
        self.rate = rospy.Rate(10)  # Publishing rate in Hz
    
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def get_distance(self, pose1, pose2):
        return math.sqrt((pose1.position.x - pose2.position.x) ** 2 +
                         (pose1.position.y - pose2.position.y) ** 2 +
                         (pose1.position.z - pose2.position.z) ** 2)

    def publish_distance(self, distance, topic):
        dist_msg = Float32()
        dist_msg.data = distance
        topic.publish(dist_msg)
    
    def publish_pose(self, pose, topic):
        pose_msg = PoseStamped()
        pose_msg.pose = pose
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        topic.publish(pose_msg)

    def run(self):
        rospy.loginfo(f"Starting Waypoint Navigator for {self.drone_name}")
        while not rospy.is_shutdown():
            if self.current_pose:
                # Publish distance from current pose to start on /if_started
                distance_to_start = self.get_distance(self.start_pose, self.current_pose)
                # rospy.loginfo(f"Distance to start: {distance_to_start:.2f}")
                self.publish_distance(distance_to_start, self.start_dist_pub)

                if self.current_waypoint_index < len(self.waypoints):
                    current_waypoint = self.waypoints[self.current_waypoint_index]
                    distance_to_waypoint = self.get_distance(current_waypoint, self.current_pose)

                    # TODO: this number is increased specifically for fast planner due to bug
                    # TODO: to reduce to a smaller number
                    if distance_to_waypoint < 2.0:  # Threshold to switch to next waypoint
                        # rospy.loginfo(f"Reaced waypoint {self.current_waypoint_index}")
                        self.current_waypoint_index += 1
                    else:
                        # rospy.loginfo(f"Distance to waypoint {self.current_waypoint_index}: {distance_to_waypoint:.2f}")
                        self.publish_pose(current_waypoint, self.waypoint_pub)
                else:
                    # Publish distance to goal on /if_reached when waypoints are exhausted
                    distance_to_goal = self.get_distance(self.goal_pose, self.current_pose)
                    # rospy.loginfo(f"Distance to goal: {distance_to_goal:.2f}")
                    self.publish_distance(distance_to_goal, self.end_dist_pub)
                    self.publish_pose(self.goal_pose, self.waypoint_pub)

            self.rate.sleep()

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Waypoint Navigator')
    parser.add_argument('--start_x', type=float, default=0.0, help='Starting X coordinate')
    parser.add_argument('--start_y', type=float, default=0.0, help='Starting Y coordinate')
    parser.add_argument('--start_z', type=float, default=1.0, help='Starting Z coordinate')
    parser.add_argument('--goal_x', type=float, default=30.0, help='Goal X coordinate')
    parser.add_argument('--goal_y', type=float, default=5.0, help='Goal Y coordinate')
    parser.add_argument('--goal_z', type=float, default=1.0, help='Goal Z coordinate')
    parser.add_argument('--drone_name', type=str, default='/quad_0', help='Drone name')
    parser.add_argument('--waypoints', type=str, nargs='+', help='List of intermediate waypoints as x y z groups')

    args = parser.parse_args()

    node_name = args.drone_name[1:] + "_waypoint_navigator"
    rospy.init_node(node_name)

    # Build the start and goal poses
    start_pose = Pose()
    start_pose.position.x = args.start_x
    start_pose.position.y = args.start_y
    start_pose.position.z = args.start_z

    goal_pose = Pose()
    goal_pose.position.x = args.goal_x
    goal_pose.position.y = args.goal_y
    goal_pose.position.z = args.goal_z

    # Build the list of waypoints (if provided)
    # TODO: Check if the logic is sound
    waypoints = []
    if args.waypoints and len(args.waypoints) % 3 == 0:
        for i in range(0, len(args.waypoints), 3):
            waypoint_pose = Pose()
            waypoint_pose.position.x = float(args.waypoints[i])
            waypoint_pose.position.y = float(args.waypoints[i + 1])
            waypoint_pose.position.z = float(args.waypoints[i + 2])
            waypoints.append(waypoint_pose)
        print(waypoints)
    else:
        rospy.logwarn("No waypoints provided or waypoints are incomplete.")

    navigator = WaypointNavigator(start_pose, goal_pose, waypoints, args.drone_name)
    navigator.run()
