#!/bin/bash

xhost +

if (($# != 3 )); then
    echo "Error: Usage ./run_data_collection.sh <planner> <test_case> <num_drones>"
    exit 1
fi

declare planner=$1
declare test_case=$2
declare num_drones=$3

if [ $num_drones -gt 8 ]; then
    echo "too many agents! max is 8"
    exit 1
fi

args="$planner $test_case $num_drones"
word=""
for arg in $args; do
    word+="\\\"$arg\\\", "
done

args="$num_drones"
num=""
for arg in $args; do
    num+="\\\"$arg\\\""
done

# Update line 4 in override,yml with the new test case
sed -i "4s/command: \[.*/command: [${word}]/" "override.yml"

sed -i "6s/command: \[.*/command: [${num}]/" "override.yml"

# Run the simulation environment and planner
docker compose -f docker-compose.yml -f override.yml up 