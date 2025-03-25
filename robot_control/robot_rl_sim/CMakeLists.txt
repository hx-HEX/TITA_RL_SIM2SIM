cmake_minimum_required(VERSION 3.16)
project(robot_rl_sim)

add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")
find_package(gazebo REQUIRED)

find_package(TBB REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(joint_state_broadcaster REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(rclcpp REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(std_srvs REQUIRED)



# Make sure to find correct Python version based on ROS distro
if(DEFINED ENV{ROS_DISTRO})
    set(ROS_DISTRO_ENV $ENV{ROS_DISTRO})
    if(ROS_DISTRO_ENV STREQUAL "foxy")
      find_package(Python3 3.8 EXACT COMPONENTS Interpreter Development REQUIRED)
    elseif(ROS_DISTRO_ENV STREQUAL "humble")
      find_package(Python3 3.10 EXACT COMPONENTS Interpreter Development REQUIRED)
    else()
      # Not very sure what would the default case be. A fuzzy match?
      find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    endif()
endif()

link_directories(/usr/local/lib)
include_directories(${YAML_CPP_INCLUDE_DIR})

include_directories(
  include
  include/robot_rl_sim
)

add_library(robot_rl_sdk src/robot_rl_sdk.cpp)
target_link_libraries(robot_rl_sdk "${TORCH_LIBRARIES}" Python3::Python Python3::Module)
set_property(TARGET robot_rl_sdk PROPERTY CXX_STANDARD 14)
find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
  target_link_libraries(robot_rl_sdk Python3::NumPy)
else()
  target_compile_definitions(robot_rl_sdk WITHOUT_NUMPY)
endif()

add_library(observation_buffer src/observation_buffer.cpp)
target_link_libraries(observation_buffer "${TORCH_LIBRARIES}")
set_property(TARGET observation_buffer PROPERTY CXX_STANDARD 14)

add_executable(robot_rl_sim src/robot_rl_sim.cpp )
target_link_libraries(robot_rl_sim
  -pthread
  robot_rl_sdk observation_buffer yaml-cpp
)

ament_target_dependencies(robot_rl_sim
  joint_state_broadcaster
  robot_state_publisher
  rclcpp
  gazebo_ros
  std_msgs
  rclpy
  gazebo_msgs
  std_srvs
)

install(TARGETS
  robot_rl_sim
  robot_rl_sdk
  observation_buffer
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch worlds models
  DESTINATION share/${PROJECT_NAME}/
)
ament_package()
