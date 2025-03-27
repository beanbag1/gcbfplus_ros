#!/bin/bash

python3.10 -m venv ~/ros_ws/src/gcbfplus_ros/.venv

source ~/ros_ws/src/gcbfplus_ros/.venv/bin/activate

pip install --upgrade pip

pip install --no-cache-dir "jax[cuda12]"

pip install --no-cache-dir -r docker/requirements.txt 