#!/bin/bash
set -e

# Add the user's local bin directory to the PATH.
# This ensures any executables installed via "pip install --user" can be found.
export PATH="/home/docker_user/.local/bin:${PATH}"

# Source the main ROS 2 setup file
source /opt/ros/${ROS_DISTRO}/setup.bash

# Source the local workspace's setup file, if it exists
if [ -f /home/docker_user/ros2_ws_jetson/install/setup.bash ]; then
  source /home/docker_user/ros2_ws_jetson/install/setup.bash
  echo "Sourced local workspace: /home/docker_user/ros2_ws_jetson"
else
  # This warning is important! The launch will fail if the workspace isn't built.
  echo "Warning: Your workspace is not sourced. Please build it first with 'colcon build'."
fi

# --- Main Logic ---
# Check if any command was passed to the entrypoint (e.g., "bash", "ls", etc.)
# The special variable "$@" holds all arguments passed to the script.
if [ $# -gt 0 ]; then
    # If arguments were provided, execute them.
    # This allows you to run "docker-compose exec ros2_dev bash" to get a shell.
    echo "Executing command: $@"
    exec "$@"
else
    # If no command was provided, run our default ROS 2 launch file.
    # The 'exec' command replaces the shell process with the ros2 launch process.
    echo "No command provided. Starting default JETSON VISION bringup launch file..."
    exec ros2 launch head_bringup head_bringup_launch.py
fi