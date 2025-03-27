import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import numpy as np
np.float = np.float64
import ros_numpy
from std_msgs.msg import Float32MultiArray
from quadrotor_msgs.msg import PositionCommand
from nav_msgs.msg import Odometry, Path
import globals
import jax.numpy as jnp
import point_cloud_utils as pcu

# pub = rospy.Publisher("quad_0/planning/pos_cmd", PositionCommand, queue_size=50)

class ROSDrone:
    def __init__(self, lidar, odom, vel, goal):
        self.lidar_data = lidar
        self.odom_data = odom
        self.vel_data = vel
        self.goal_data = goal
        self.initialised = False
        self.new_data = False

last_callback = 0

drone_dict = {}

def lidar_callback(data, arg):
    global last_callback
    start = rospy.get_time()
    time_since_previous = start - last_callback
    # print(f"time since last callback: {time_since_previous}")
    last_callback = rospy.get_time()
    pc = ros_numpy.numpify(data)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']

    states = drone_dict[f"quad_{arg}"].odom_data

    # maybe use voxel grid to downsample
    idx = pcu.downsample_point_cloud_poisson_disk(points, 0.5) 
    points = points[idx]
    time_elapsed = (rospy.get_time() - start)*1000
    # print(f"downsample: {time_elapsed}")
    # num_voxels_per_axis = 256
    # bbox_size = points.max(0) - points.min(0)
    # sizeof_voxel = bbox_size / num_voxels_per_axis
    # points = pcu.downsample_point_cloud_on_voxel_grid(sizeof_voxel, points)

    rm_list = np.zeros(1, dtype=int)
    for i in range(len(points)):
        temp = points[i]
        if (temp[2] < 0.5):
            # print(temp[2])
            rm_list = np.append(rm_list, i)
    rm_list = np.delete(rm_list, 0)
    points = np.delete(points, rm_list, axis=0)
    
    distance_list = np.empty(1)
    for i in range(len(points)):
        temp = points[i]
        temp = np.subtract(temp, states[0])
        distance = temp[0]**2 + temp[1]**2 + temp[2]**2
        distance_list = np.append(distance_list, distance)
    time_elapsed = (rospy.get_time() - start)*1000
    # print(f"dist calc: {time_elapsed}")
    distance_list = np.delete(distance_list, 0)
    num_rays = np.min([len(points) - 1, 32])
    indices = globals.find_n_smallest_indices(distance_list, num_rays)
    time_elapsed = (rospy.get_time() - start)*1000
    # print(f"callback: {time_elapsed}")
    globals.lidar_lock = True
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(np.append(points[indices], [[7, 6, 2.5]], axis=0))
    # point_cloud.points = o3d.utility.Vector3dVector(points[indices])
    # o3d.visualization.draw_geometries([point_cloud], mesh_show_wireframe=True)
    drone_dict[f"quad_{arg}"].lidar_data = points[indices]
    drone_dict[f"quad_{arg}"].initialised = True

def odom_callback(data, arg):
    a = data.pose.pose.position
    odom_data = jnp.array([a.x, a.y, a.z], ndmin = 2)
    drone_dict[f"quad_{arg}"].odom_data = odom_data
    drone_dict[f"quad_{arg}"].new_data = True

def vel_callback(data, arg):
    a = data.data
    vel_data = jnp.array([a[0], a[1], a[2]], ndmin = 2)
    drone_dict[f"quad_{arg}"].vel_data = vel_data

def goal_callback(data, arg):
    goal_data = jnp.array([data.pose.position.x, data.pose.position.y, 0, 0], ndmin=2)
    drone_dict[f"quad_{arg}"].goal_data = goal_data

def send_control(control, odom_data, curr_vel, dt, factor, id):
    # print(id)
    vel_max = 1.0
    dt = np.minimum(dt, 0.100)
    pub_msg = PositionCommand()
    pub_msg.position.x = (odom_data[0] + curr_vel[0]*dt + 0.5*control[0]*(dt**2))*factor
    pub_msg.position.y = (odom_data[1] + curr_vel[1]*dt + 0.5*control[1]*(dt**2))*factor
    pub_msg.position.z = 2
    pub_msg.velocity.x = jnp.clip((curr_vel[0] + control[0]*dt)*factor, -vel_max, vel_max)
    pub_msg.velocity.y = jnp.clip((curr_vel[1] + control[1]*dt)*factor, -vel_max, vel_max)
    # pub_msg.velocity.z = jnp.clip((curr_vel[2] + control[2]*dt)*factor, -vel_max, vel_max)
    # print(pub_msg.velocity)
    control_clip_high = np.minimum(control*factor, 0.0)
    control_clip_low = np.maximum(control*factor, 0.0)
    final_control_x = control[0]*factor
    if (curr_vel[0]*factor >= vel_max):
        final_control_x = control_clip_high[0]
    elif (curr_vel[0]*factor <= -vel_max):
        final_control_x = control_clip_low[0]
    final_control_y = control[1]*factor 
    if (curr_vel[1]*factor >= vel_max):
        final_control_y = control_clip_high[1]
    elif (curr_vel[1]*factor <= -vel_max):
        final_control_y = control_clip_low[1]
    # final_control_z = control[2]*factor 
    # if (curr_vel[2]*factor >= vel_max):
    #     final_control_z = control_clip_high[2]
    # elif (curr_vel[2]*factor <= -vel_max):
    #     final_control_z = control_clip_low[2]
    pub_msg.acceleration.x = final_control_x
    pub_msg.acceleration.y = final_control_y
    # print(pub_msg.acceleration)
    # pub_msg.acceleration.z = final_control_z
    drone_dict[f"quad_{id}_pub"].publish(pub_msg)

def get_data():
    global new_lidar
    # process the data
    num_agents = int(len(drone_dict)/2)
    lidar_data = np.zeros((1, 3))
    odom_data = np.zeros((1, 3))
    vel_data = np.zeros((1, 3))
    goal_data = np.zeros((1, 4))
    for id in range(num_agents):
        while not drone_dict[f"quad_{id}"].initialised:
            pass
        agent_lidar_data = drone_dict[f"quad_{id}"].lidar_data
        lidar_data = np.append(lidar_data, agent_lidar_data, axis=0)
        while not drone_dict[f"quad_{id}"].new_data:
            pass
        agent_odom_data = drone_dict[f"quad_{id}"].odom_data
        odom_data = np.append(odom_data, agent_odom_data, axis=0)
        drone_dict[f"quad_{id}"].new_data = False
        agent_vel_data = drone_dict[f"quad_{id}"].vel_data
        vel_data = np.append(vel_data, agent_vel_data, axis=0)
        agent_goal_data = drone_dict[f"quad_{id}"].goal_data
        goal_data = np.append(goal_data, agent_goal_data, axis=0)
    lidar_data = np.delete(lidar_data, 0, axis=0)
    odom_data = np.delete(odom_data, 0, axis=0)
    vel_data = np.delete(vel_data, 0, axis=0)
    goal_data = np.delete(goal_data, 0, axis=0)
    new_lidar = 0
    return lidar_data, odom_data, vel_data, goal_data

def init_ros_interface(num_agents):
    rospy.init_node('gcbfplus_ros', anonymous=True)
    for id in range(num_agents):
        rospy.Subscriber(f"quad{id}_pcl_render_node/cloud", PointCloud2, lidar_callback, id)
        rospy.Subscriber(f"quad_{id}/odom", Odometry, odom_callback, id)
        rospy.Subscriber(f"quad_{id}/current_vel", Float32MultiArray, vel_callback, id)
        rospy.Subscriber(f"quad_{id}/goal", PoseStamped, goal_callback, id)
        drone_dict[f"quad_{id}"] = ROSDrone(np.zeros((1, 3)), jnp.zeros((1, 3)), jnp.array([0, 0, 0], ndmin = 2), jnp.zeros((1, 4)))
        drone_dict[f"quad_{id}_pub"] = rospy.Publisher(f"quad_{id}/planning/pos_cmd", PositionCommand, queue_size=50)
    # rospy.Timer(rospy.Duration(0.05), talker_callback)
    rospy.sleep(1)
    rospy.loginfo("started")

# rospy.init_node('gcbfplus_ros', anonymous=True)
# rospy.Subscriber("quad0_pcl_render_node/cloud", PointCloud2, lidar_callback)
# rospy.loginfo("lidar listener started")
# rospy.Subscriber("quad0/odom", Odometry, odom_callback)
# rospy.loginfo("odometry listener started")
# pub = rospy.Publisher("quad_0/planning/pos_cmd", PositionCommand, queue_size=50)
# rospy.Timer(rospy.Duration(0.05), talker_callback)
# rospy.loginfo("poscmd talker started")
# print("hello")


# rospy.init_node('listener', anonymous=True)

# while not rospy.is_shutdown():
#     str = "hello world %s"%rospy.get_time()
#     rospy.loginfo(str)
#     rospy.sleep(1)

# rospy.loginfo("shutting down")