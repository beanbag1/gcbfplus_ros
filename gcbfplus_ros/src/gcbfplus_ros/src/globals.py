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
lidar_lock = False

def find_n_smallest_indices(arr, n):
    indices = np.empty(1, int)
    try:
        p_arr = np.partition(arr, n)
        for x in range(n):
            for y in range(len(arr)):
                if p_arr[x] == arr[y]:
                    indices = np.append(indices, y)
        indices = np.delete(indices, 0)
    except:
        pass
    return indices