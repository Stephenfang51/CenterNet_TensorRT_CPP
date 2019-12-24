cmake_minimum_required(VERSION 3.15)
project(centerNet_trt)


set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
message("OpenCV version: " ${OpenCV_VERSION})
#openCV
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR}/include)
link_directories(${OpenCV_LIB_DIR})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}./src src)
#set source file
#set(src main.cpp include/argparse.h src/argparse.cpp)
set(src ${src} main.cpp example/demo.cpp include/ct_trt_config.h include/cttrt_Net.h include/utils.h)

add_executable(${PROJECT_NAME} ${src})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})