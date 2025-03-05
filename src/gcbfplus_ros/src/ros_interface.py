import rospy
from sensor_msgs.msg import PointCloud2
import geometry_msgs
import numpy as np
np.float = np.float64
import ros_numpy
from std_msgs.msg import Float32MultiArray
from quadrotor_msgs.msg import PositionCommand
from nav_msgs.msg import Odometry
import globals
import jax.numpy as jnp

pub = rospy.Publisher("quad_0/planning/pos_cmd", PositionCommand, queue_size=50)

def lidar_callback(data):
    # if globals.lidar_lock: 
    #     pass
    # else:
    #     pc = ros_numpy.numpify(data)
    #     points=np.zeros((pc.shape[0],3))
    #     points[:,0]=pc['x']
    #     points[:,1]=pc['y']
    #     points[:,2]=pc['z']
    #     globals.new_lidar_data = points
    #     globals.new_num_rays = len(points)
    #     print(globals.num_rays)
    pc = ros_numpy.numpify(data)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    # remove_list = np.array([0])
    # print("ahhhh")
    # for i in range(len(points)):
    #     temp = points[i]
    #     distance = np.sqrt(temp[0]**2 + temp[1]**2 + temp[2]**2)
    #     if distance > 2:
    #         remove_list = np.append(remove_list, [i])
    # np.delete(points, remove_list)
    globals.new_lidar_data = points[::50000, ::]
    globals.new_num_rays = len(globals.new_lidar_data)

def odom_callback(data):
    a = data.pose.pose.position
    globals.odom_data = jnp.array([a.x, a.y, a.z], ndmin = 2)

def vel_callback(data):
    a = data.data
    globals.vel_data = jnp.array([a[0], a[1], a[2]], ndmin = 2)

def talker_callback(event):
    pub_msg = PositionCommand()
    pub_msg.position.x = globals.odom_data[0][0] + globals.control_vector[0]*3
    pub_msg.position.y = globals.odom_data[0][1] + globals.control_vector[1]*3
    pub_msg.position.z = globals.odom_data[0][2] + globals.control_vector[2]*3
    # pub_msg.position.z = 2.5
    pub.publish(pub_msg)

def init_ros_interface():
    rospy.init_node('gcbfplus_ros', anonymous=True)
    rospy.Subscriber("quad0_pcl_render_node/sensor_cloud", PointCloud2, lidar_callback)
    rospy.loginfo("lidar listener started")
    rospy.Subscriber("quad_0/odom", Odometry, odom_callback)
    rospy.loginfo("odometry listener started")
    rospy.Subscriber("quad_0/current_vel", Float32MultiArray, vel_callback)
    pub = rospy.Publisher("quad_0/planning/pos_cmd", PositionCommand, queue_size=50)
    rospy.Timer(rospy.Duration(0.05), talker_callback)
    rospy.loginfo("poscmd talker started")
    print("hello")

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