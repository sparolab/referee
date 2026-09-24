#!/bin/bash
# Sets up the ROS 2 environment, the workspace overlay when one has been built,
# and a few helpers, then hands over to the command.
set -e

source /opt/ros/humble/setup.bash
if [ -f "$REFEREE_ROS_INSTALL/setup.bash" ]; then
  source "$REFEREE_ROS_INSTALL/setup.bash"
fi

# Rebuild the C++ core against the mounted sources.
referee-build-core() {
  cmake -S /referee/cpp/referee -B "$REFEREE_CORE_BUILD" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$REFEREE_CORE_BUILD" -j"$(nproc)"
}

# Build the ROS 2 workspace, if one is mounted.
referee-build() {
  if [ ! -d /referee/ros2_ws/src ]; then
    echo "no /referee/ros2_ws/src - nothing to build" >&2
    return 0
  fi
  # colcon writes its log next to the working directory, and every path it is
  # given must be outside the read-only source mount.
  cd /opt/referee
  colcon --log-base "$REFEREE_ROS_LOG" build --symlink-install \
    --base-paths /referee/ros2_ws/src \
    --build-base "$REFEREE_ROS_BUILD" \
    --install-base "$REFEREE_ROS_INSTALL" \
    --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DREFEREE_BUILD_TOOLS=OFF
  source "$REFEREE_ROS_INSTALL/setup.bash"
}

# Run every check against the mounted datasets.
referee-validate() {
  bash /referee/docker/validate.sh "$@"
}

export -f referee-build-core referee-build referee-validate
exec "$@"
