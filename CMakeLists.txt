project(test)
cmake_minimum_required(VERSION 3.4)

SET(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
#find_package( Sophus REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories( ${OpenCV_DIRS} )
include_directories( /usr/include/eigen3/ )
include_directories( /opt/dep/eigen3/include/eigen3 )
include_directories( ${Sophus_INCLUDE_DIRS} )
include_directories( dep/sophus/include )
include_directories( ./mvslam )

FILE(GLOB  SRC_ALL mvslam/*.cpp)
add_executable(vo ${SRC_ALL})
target_link_libraries(vo   ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES})

