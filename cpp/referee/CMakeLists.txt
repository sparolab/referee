cmake_minimum_required(VERSION 3.16)
project(referee LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# The descriptors and the lookup are header-only; only the reader and the
# feature extractor need compiling. No ROS, no Boost, no PCL.
add_library(referee src/radar_utils.cpp src/features.cpp)
target_include_directories(referee PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(referee PUBLIC ${OpenCV_LIBS} Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(referee PUBLIC OpenMP::OpenMP_CXX)
endif()

option(REFEREE_BUILD_TOOLS "Build validation/CLI tools" ON)
if(REFEREE_BUILD_TOOLS)
  add_executable(validate_descriptor tools/validate_descriptor.cpp)
  target_link_libraries(validate_descriptor PRIVATE referee)
endif()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS referee EXPORT refereeTargets)
