project(rostest)
cmake_minimum_required(VERSION 3.4)

SET(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)


## Build talker and listener
include_directories( 
    ${catkin_INCLUDE_DIRS} 
    ./  
    ./mvslam/  
    )

add_executable(mvslam_node 
    mvslam_node.cpp
    ./mvslam/vop.cpp
    )

target_link_libraries(mvslam_node 
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    )


