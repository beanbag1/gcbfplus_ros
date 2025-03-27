# MARSIM
MARSIM: A light-weight point-realistic simulator for LiDAR-based UAVs

Paper is available on Arxiv: https://arxiv.org/abs/2211.10716

The video is available on youtube: https://youtu.be/hiRtcq-5lN0

## Update

Ubuntu 20.04 is also supported in ubuntu20 branch.

**Ten realistic maps (low and high resolution) have been realeased in the realease packages.**

**A new branch that merge with FUEL has been released in the fuel_ubuntu20 branch.**

## Prerequisited

### Ubuntu and ROS

Ubuntu 16.04~20.04.  [ROS Installation](http://wiki.ros.org/ROS/Installation).

### PCL && Eigen && glfw3

PCL>=1.6, Follow [PCL Installation](https://pointclouds.org/). 

Eigen>=3.3.4, Follow [Eigen Installation](https://eigen.tuxfamily.org/index.php?title=Main_Page).

glfw3:
```
sudo apt-get install libglfw3-dev libglew-dev
```

### make
```
mkdir -p marsim_ws/src
cd marsim_ws/src
git clone git@github.com:hku-mars/MARSIM.git
cd ..
catkin_make
```

## run the simulation

```
source devel/setup.bash
roslaunch test_interface single_drone_avia.launch
```
Click on 3Dgoal tool on the Rviz, you can give the UAV a position command to control its flight.

For now, we provide several launch files for users, which can be found in test_interface/launch folder.

You can change the parameter in launch files to change the map and LiDAR to be simulated.

** If you want to use the GPU version of MARSIM, please set the parameter "use_gpu" to true. **

## run the simulation with FUEL algorithm

You should first change the branch to fuel_ubuntu20 branch. If you are using ubuntu 20.04, you should first download Nlopt and make install it in your environment. Then you can run the simulation by the command below:
```
source devel/setup.bash
roslaunch exploration_manager exploration.launch
```
Then click on 2Dgoal tool on the Rviz, randomly click on the map, and FUEL would automously run.

## Running with FAST Planner
1. Clone the private FAST Planner repo
```
git clone -b development git@github.com:DinoHub/fast_planner.git
cd docker
chmod +x run_docker.sh
./run_docker.sh
# within docker
cd /root/ros_ws
catkin_make
```
2. Run the simulation
```
roslaunch test_interface single_drone_test.launch
```
3. In Docker container, start FAST Planner
```
roslaunch plan_manage kino_replan_marsim.launch
```
4. In Docker container, publish to the ```/goal``` topic the desired goal
```
rostopic pub /goal geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: 0.0
    y: 0.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0"

```
## Customisation
To change the sensor types, you can change the ```single_drone_test.launch``` Currently only os128, avia, mid360 lidars are supported. To create custom sensors, create the sensor configs in ```test_interface/config``` and update ```single_drone_refactored.xml``` accordingly. A sample config file is shown below for the ouster128 lidar
```
name: quad$(arg drone_id)_pcl_render_node
  drone_id: $(arg drone_id)
  quadrotor_name: quad_$(arg drone_id)
  uav_num: $(arg uav_num_)
  is_360lidar: 1
  sensing_horizon: 15.0
  sensing_rate: 10.0
  estimation_rate: 10.0
  polar_resolution: 0.2
  yaw_fov: 360.0
  vertical_fov: 90.0
  min_raylength: 1.0
  livox_linestep: 1.4
  curvature_limit: 100.0
  hash_cubesize: 5.0
  use_avia_pattern: 0
  use_vlp32_pattern: 0
  use_minicf_pattern: 1
  downsample_res: $(arg downsample_resolution_)
  dynobj_enable: 0
  dynobject_size: 0.6
  dynobject_num: 30
  dyn_mode: 2
  dyn_velocity: 3.0
  use_uav_extra_model: $(arg use_uav_extra_model_)
  collisioncheck_enable: 1
  collision_range: 0.05
  output_pcd: 0
```

## Acknowledgments
Thanks for [FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL.git)

## Future
More realistic maps and functions are going to be released soon.
