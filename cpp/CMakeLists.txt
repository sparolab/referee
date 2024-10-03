project(referee)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wno-unused-parameter -O2")

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(OMP_NUM_THREADS 8)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS
    system
    filesystem
)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_library(features src/features.cpp)
target_link_libraries(features
    ${OpenCV_LIBRARIES}
    ${Boost_INCLUDE_DIRS}
    Eigen3::Eigen
)

add_library(referee src/referee.cpp)
target_link_libraries(referee
    ${Boost_INCLUDE_DIRS}
    Eigen3::Eigen
)

add_library(radar_utils src/radar_utils.cpp)
target_link_libraries(radar_utils
	${OpenCV_LIBRARIES}
	${Boost_INCLUDE_DIRS}
  Eigen3::Eigen
)

add_executable(test_referee src/main.cpp)
target_link_libraries(test_referee   
    features
    referee
    radar_utils
)