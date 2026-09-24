# ReFeree — ROS 2 Humble (Ubuntu 22.04)
#
# Carries everything the project needs: the header-only C++ core (OpenCV only),
# the Python reference used to validate it, and the heavier stack the SLAM side
# wants — GTSAM, PCL, FFTW — so odometry and LT-SLAM build in the same image.
FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget \
        libboost-all-dev libeigen3-dev \
        libopencv-dev \
        libfftw3-dev libfftw3-bin \
        libtbb-dev libomp-dev \
        python3-pip python3-numpy python3-scipy \
        python3-colcon-common-extensions \
        ros-humble-cv-bridge ros-humble-image-transport \
        ros-humble-pcl-ros ros-humble-pcl-conversions \
        ros-humble-tf2 ros-humble-tf2-ros ros-humble-tf2-eigen \
        ros-humble-tf2-geometry-msgs \
        ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# GTSAM 4.2, C++17, system Eigen so it agrees with PCL and the ROS stack.
RUN git clone --depth 1 --branch 4.2 https://github.com/borglab/gtsam.git /opt/gtsam && \
    cmake -S /opt/gtsam -B /opt/gtsam/build \
        -DGTSAM_USE_SYSTEM_EIGEN=ON \
        -DGTSAM_BUILD_TESTS=OFF \
        -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
        -DGTSAM_WITH_TBB=OFF \
        -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/gtsam/build -j"$(nproc)" --target install && \
    ldconfig && rm -rf /opt/gtsam/build

WORKDIR /referee

# Build trees live outside /referee so they never collide with the bind-mounted
# sources, and so a container run leaves no root-owned files in the host repo.
ENV REFEREE_ROOT=/referee \
    REFEREE_CORE_BUILD=/opt/referee/core-build \
    REFEREE_ROS_BUILD=/opt/referee/ros-build \
    REFEREE_ROS_INSTALL=/opt/referee/ros-install \
    REFEREE_ROS_LOG=/opt/referee/ros-log

# The C++ core is self-contained, so it builds before anything ROS-shaped and
# its validation tools are available even when the workspace is empty.
COPY cpp /referee/cpp
COPY python /referee/python
RUN cmake -S /referee/cpp/referee -B "$REFEREE_CORE_BUILD" \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build "$REFEREE_CORE_BUILD" -j"$(nproc)"

# The ROS 2 workspace is mounted at run time (see docker-compose.yml) so host
# edits take effect without a rebuild; `referee-build` compiles it.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
