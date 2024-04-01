#!/usr/bin/env bash
# This script will build a Docker image with orbbec_ros in a catkin workspace,
# then run it with host networking and the sources mounted read-only.

cd "$(dirname "${BASH_SOURCE[0]}")"

CATKIN_DIR=/catkin_ws

docker build -t orbbec_ros .
docker run -it --rm --net=host --privileged\
  -v "$(pwd)/OrbbecSDK_ROS1/:${CATKIN_DIR}/src/OrbbecSDK_ROS1:ro" \
  -v /etc/udev/*:/etc/udev/* \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "/dev/video*:/dev/video*:rw" \
  -v /dev:/dev \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  orbbec_ros $@
