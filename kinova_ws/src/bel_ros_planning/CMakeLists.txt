cmake_minimum_required(VERSION 3.0.2)
project(bel_ros_planning)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED
             COMPONENTS
             moveit_core
             moveit_ros_planning
             moveit_ros_planning_interface
             pluginlib
             cmake_modules
             geometric_shapes
             eigen_conversions
)

find_package(Boost REQUIRED system filesystem date_time thread)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS 
    moveit_core
    moveit_ros_planning_interface
    interactive_markers
    eigen_conversions
)
find_package(Eigen REQUIRED)

find_package(kinova_driver REQUIRED)

###########
## Build ##
###########
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

include_directories(SYSTEM ${Boost_INCLUDE_DIR} ${EIGEN_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

install(PROGRAMS
  src/run_inference.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

