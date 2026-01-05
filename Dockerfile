# Use an official ROS 2 image as a parent image
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Set timezone
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies and tools as root
RUN apt-get update && apt-get install -y \
    sudo build-essential git vim v4l-utils \
    python3-colcon-common-extensions python3-rosdep python3-pip \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-camera-calibration \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    #libpcl-dev \
    ros-${ROS_DISTRO}-pcl-conversions 
    #Is pcl-conversions necessary?
RUN apt update && apt install -y \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep as root
RUN rosdep init || true && rosdep update

# Create the user
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o docker_user && \
    useradd -m -u $UID -g $GID -s /bin/bash docker_user

# Add the user to the sudo and video groups
ARG VIDEO_GID
#ARG DIALOUT_GID
RUN if [ -n "$VIDEO_GID" ]; then \
        if ! getent group $VIDEO_GID > /dev/null; then groupadd -g $VIDEO_GID video; fi; \
    fi && \
    #if [ -n "$DIALOUT_GID" ]; then \
    #    if ! getent group $DIALOUT_GID > /dev/null; then groupadd -g $DIALOUT_GID dialout; fi; \
    #fi && \
    usermod -aG sudo,video docker_user

# Give the user password-less sudo privileges
RUN echo "docker_user ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/docker-user-sudo

# Copy the entrypoint script and set its permissions AS ROOT
COPY entrypoint.sh /home/docker_user/entrypoint.sh
RUN chown docker_user:docker_user /home/docker_user/entrypoint.sh && \
    chmod +x /home/docker_user/entrypoint.sh

# NOW we switch to the new user.
USER docker_user
WORKDIR /home/docker_user

# --- THE FIX IS HERE ---
# 1. Install Python packages AS THE USER.
# The --user flag installs them into /home/docker_user/.local/


RUN python3 -m pip install --user \
    'numpy<2.0' \
    opencv-python \
    mediapipe \
    'git+https://github.com/Chandrahas-kasoju/python-st3215.git' \
    requests \
    pygame

# Create workspace directory as the user
RUN mkdir -p /home/docker_user/ros2_ws_jetson/src
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/docker_user/.bashrc
RUN echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/docker_user/.bashrc
RUN echo "source /home/docker_user/ros2_ws_jetson/install/setup.bash" >> /home/docker_user/.bashrc
RUN echo "export ROS_DOMAIN_ID=0" >> /home/docker_user/.bashrc
# Set the entrypoint
ENTRYPOINT ["/home/docker_user/entrypoint.sh"]
# Set the default command
CMD ["bash"]
