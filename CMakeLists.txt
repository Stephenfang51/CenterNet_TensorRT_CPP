cmake_minimum_required(VERSION 3.15)
project(centerNet_trt)


set(CMAKE_CXX_STANDARD 14)

#find OpenCV
find_package(OpenCV REQUIRED)
message("OpenCV version: " ${OpenCV_VERSION})



#find CUDA
find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

message("CUDA version: " ${CUDA_VERSION})

########################## TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include/)

message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSER nvparsers
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN}
        ${TENSORRT_LIBRARY_PARSER} ${TENSORRT_LIBRARY_ONNXPARSER})
message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")

find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)

if(NOT TENSORRT_FOUND)
    message(ERROR
            "Cannot find TensorRT library.")
endif()

#########################  end TensorRT




#指出所有的头文件 包含openCV CUDA CUDNN include
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/include )
link_directories(${OpenCV_LIB_DIR})



#指出所有源文件
file(GLOB CPP_SRC src/*.cpp) ## src下所有的cpp文件
#file(GLOB CU_SRC src/*.cu) ## src下cu文件
#aux_source_directory(${CMAKE_CURRENT_LIST_DIR}./src src)
#set source file
#set(src main.cpp include/argparse.h src/argparse.cpp)

#set(src ${src}  example/demo.cpp)
#src先将测试用的main.cpp拿掉， 留下demo.cpp中的main 否则重叠

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(src ${src}  example/demo.cpp)


#add_executable(main_test ${src})
add_executable(demo example/demo.cpp ${CPP_SRC})
target_link_libraries(demo ${OpenCV_LIBS} ${TENSORRT_LIBRARY})


#add_executable(demo example/demo.cpp ${src})
#target_link_libraries(demo ${OpenCV_LIBS})