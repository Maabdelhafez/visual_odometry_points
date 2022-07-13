project(test)
cmake_minimum_required(VERSION 3.4)

SET(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
#find_package( Sophus REQUIRED )

include_directories( ${OpenCV_DIRS} )
include_directories( /usr/include/eigen3/ )
include_directories( ${Sophus_INCLUDE_DIRS} )
include_directories( dep/sophus/include )

FILE(GLOB  SRC_ALL *.cpp)
add_executable(vo ${SRC_ALL})
target_link_libraries(vo   ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES})

