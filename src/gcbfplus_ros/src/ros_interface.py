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
import open3d as o3d

# pub = rospy.Publisher("quad_0/planning/pos_cmd", PositionCommand, queue_size=50)
lidar_data = np.zeros(1)
odom_data = jnp.zeros((1, 3))
vel_data = jnp.array([0, 0, 0], ndmin = 2)
goal_data = jnp.array([0, 0, 0, 0], ndmin = 2)

last_callback = 0

def lidar_callback(data):
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

    states = globals.odom_data

    # print(len(points))
    # rm_list = np.zeros(1, dtype=int)
    # for i in range(len(points)):
    #     temp = points[i]
    #     if np.abs(temp[2] - states[0][2]) > 0.2:
    #         # print(temp[2])
    #         rm_list = np.append(rm_list, i)
    # rm_list = np.delete(rm_list, 0)
    # points = np.delete(points, rm_list, axis=0)

    time_elapsed = (rospy.get_time() - start)*1000
    # print(f"transfer: {time_elapsed}")
    # maybe use voxel grid to downsample
    idx = pcu.downsample_point_cloud_poisson_disk(points, 0.5) 
    points = points[idx]
    # num_voxels_per_axis = 256
    # bbox_size = points.max(0) - points.min(0)
    # sizeof_voxel = bbox_size / num_voxels_per_axis
    # points = pcu.downsample_point_cloud_on_voxel_grid(sizeof_voxel, points)
    
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
    globals.new_lidar_data = points[indices]
    globals.new_num_rays = len(globals.new_lidar_data)
    time_elapsed = (rospy.get_time() - start)*1000
    # print(f"callback: {time_elapsed}")
    globals.lidar_lock = True
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(np.append(points[indices], [[7, 6, 2.5]], axis=0))
    # point_cloud.points = o3d.utility.Vector3dVector(points[indices])
    # o3d.visualization.draw_geometries([point_cloud], mesh_show_wireframe=True)

def odom_callback(data):
    a = data.pose.pose.position
    globals.odom_data = jnp.array([a.x, a.y, a.z], ndmin = 2)
    global odom_data
    odom_data = jnp.array([a.x, a.y, a.z], ndmin = 2)
    # print(globals.odom_data)

def vel_callback(data):
    a = data.data
    globals.vel_data = jnp.array([a[0], a[1], a[2]], ndmin = 2)
    global vel_data
    vel_data = jnp.array([a[0], a[1], a[2]], ndmin = 2)

def goal_callback(data):
    globals.goal[0][0] = data.pose.position.x
    globals.goal[0][1] = data.pose.position.y
    global goal_data
    goal_data = jnp.array([data.pose.position.x, data.pose.position.y, 0, 0])

# def talker_callback(event):
#     pub_msg = PositionCommand()
#     pub_msg.position.x = globals.odom_data[0][0] + globals.control_vector[0]*2
#     pub_msg.position.y = globals.odom_data[0][1] + globals.control_vector[1]*2
#     # pub_msg.position.z = globals.odom_data[0][2] + globals.control_vector[2]*0.5
#     pub_msg.position.z = 1
#     pub.publish(pub_msg)

def send_control(control, curr_vel, dt):
    factor = 2
    pub_msg = PositionCommand()
    pub_msg.position.x = globals.odom_data[0][0] + (curr_vel[0]*dt + 0.5*control[0]*(dt**2))*factor
    pub_msg.position.y = globals.odom_data[0][1] + (curr_vel[1]*dt + 0.5*control[1]*(dt**2))*factor
    pub_msg.position.z = 2
    pub_msg.velocity.x = (curr_vel[0] + control[0]*dt)*factor
    pub_msg.velocity.y = (curr_vel[1] + control[1]*dt)*factor
    pub_msg.acceleration.x = control[0]*factor
    pub_msg.acceleration.x = control[0]*factor
    pub0.publish(pub_msg)  

def get_data():
    return lidar_data, odom_data, vel_data, goal_data

def init_ros_interface():
    rospy.init_node('gcbfplus_ros', anonymous=True)
    rospy.Subscriber("quad0_pcl_render_node/cloud", PointCloud2, lidar_callback)
    rospy.Subscriber("quad_0/odom", Odometry, odom_callback)
    rospy.Subscriber("quad_0/current_vel", Float32MultiArray, vel_callback)
    rospy.Subscriber("/goal", PoseStamped, goal_callback)
    global pub0
    pub0 = rospy.Publisher("quad_0/planning/pos_cmd", PositionCommand, queue_size=50)
    # rospy.Timer(rospy.Duration(0.05), talker_callback)

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