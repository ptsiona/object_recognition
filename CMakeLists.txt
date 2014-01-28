project(BasicObjectRecognition)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_BUILD_TYPE "Release")

set(STD_FLAGS_DEBUG "-Wall -g")
set(STD_FLAGS_RELEASE "-Wall -export-dynamic -march=nocona -mfpmath=sse -ftree-vectorize -fextended-identifiers -O3") 

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(object_recognition object_recognition.cpp 
	segmentation.cpp
	pcl_high_level_ctl.cpp
	recognizer.cpp)
target_link_libraries(object_recognition ${PCL_LIBRARIES})
