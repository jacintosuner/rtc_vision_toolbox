ARG DOCKER_ROS_DISTRO=noetic
FROM ros:${DOCKER_ROS_DISTRO}

# Booststrap workspace.
ENV CATKIN_DIR=/catkin_ws
RUN . /opt/ros/$ROS_DISTRO/setup.sh \
 && mkdir -p $CATKIN_DIR/src \
 && cd $CATKIN_DIR/src \
 && catkin_init_workspace
WORKDIR $CATKIN_DIR

# Install dependencies first.
COPY OrbbecSDK_ROS1/package.xml $CATKIN_DIR/src/OrbbecSDK_ROS1/
#COPY orbbec/package.xml $CATKIN_DIR/src/orbbec/

RUN . /opt/ros/$ROS_DISTRO/setup.sh \
 && apt-get update \
 && rosdep update \
 && rosdep install \
    --from-paths src \
    --ignore-src \
    --rosdistro $ROS_DISTRO \
    -y \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y udev
RUN apt-get install libgflags-dev -y 
RUN apt install ros-$ROS_DISTRO-image-geometry 
RUN apt install ros-$ROS_DISTRO-camera-info-manager
RUN apt install ros-$ROS_DISTRO-image-transport 
RUN apt install ros-$ROS_DISTRO-image-publisher 
RUN apt install libgoogle-glog-dev -y
RUN apt install libusb-1.0-0-dev -y
RUN apt install libeigen3-dev -y

# Build rosbridge packages.
COPY OrbbecSDK_ROS1 $CATKIN_DIR/src/OrbbecSDK_ROS1
COPY orbbec /orbbec
RUN . /opt/ros/$ROS_DISTRO/setup.sh \
 && catkin_make

# We want the development workspace active all the time.
RUN echo "#!/bin/bash\n\
set -e\n\
source \"${CATKIN_DIR}/devel/setup.bash\"\n\
exec \"\$@\"" > /startup.sh \
 && chmod a+x /startup.sh \
 && echo "source ${CATKIN_DIR}/devel/setup.bash" >> /root/.bashrc
ENTRYPOINT ["/startup.sh"]
CMD ["bash"]
