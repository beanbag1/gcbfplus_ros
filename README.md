# GCBF+ for ROS
ROS implementation of the GCBF+ algorithm, from https://github.com/MIT-REALM/gcbfplus

## About
This repo contains an adaptation of the GCBF+ algorithm, for application to real-world vehicles, built on ROS Noetic. This implementation is meant to be used in conjunction with Flightbench and MARSIM for testing purposes. \
sample videos can be found here: https://youtube.com/playlist?list=PL12zLXi4Ok1X7GVKHpEpGT2uaTv1C64nN&si=iJ7RtSvj2UF87I7p

## Setup
ensure u have a working cuda installation

## Usage
To start running, run 
``` 
cd gcbfplus_ros/Flightbench/docker
./run_data_collection.sh <planner_name> <test_case> <num_agents>
``` 

## Known issues
sometimes it just doesn't start. simply try again