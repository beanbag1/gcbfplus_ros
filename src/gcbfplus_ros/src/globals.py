import rospy
# import pcl
from sensor_msgs.msg import PointCloud2
import geometry_msgs
import jax.numpy as jnp
import numpy as np
np.float = np.float64
import ros_numpy
from quadrotor_msgs.msg import PositionCommand

# lidar_data = pcl.PointCloud()
control_vector = np.array([0, 0, 0])
new_lidar_data = jnp.zeros((16, 3))
lidar_data = jnp.zeros((16, 3))
new_num_rays = 16
num_rays = 16
lidar_lock = False
odom_data = jnp.array([0, 0, 0], ndmin = 2)
vel_data = jnp.array([0, 0, 0], ndmin = 2)

start_time = 0
def print_time_elapsed(text=""):
    time_elapsed = (rospy.get_time() - start_time)*1000
    print(f"time elapsed: {time_elapsed}ms {text}")