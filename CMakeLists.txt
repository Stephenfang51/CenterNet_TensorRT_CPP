cmake_minimum_required(VERSION 3.5)
project(centerNet_trt)

#设定好库生成的路径
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib) #静态
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib) #动态
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE Release)
set(GPU_ARCHS 53)  ## config your GPU_ARCHS,See [here](https://developer.nvidia.com/cuda-gpus) for finding what maximum compute capability your specific GPU supports.
set(TENSORRT_ROOT /usr/include/aarch64-linux-gnu)

#find OpenCV
#find_package(OpenCV REQUIRED)
#message("OpenCV version: " ${OpenCV_VERSION})



#find CUDA
find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)
message("CUDA version: " ${CUDA_VERSION})



# find TensorRT
#find_path(TENSORRT_INCLUDE_DIR NvInfer.h
#        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
#        PATH_SUFFIXES include/)
#
#message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
#find_library(TENSORRT_LIBRARY_INFER nvinfer
#        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#        PATH_SUFFIXES lib lib64 lib/x64)
#find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
#        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#        PATH_SUFFIXES lib lib64 lib/x64)
#find_library(TENSORRT_LIBRARY_PARSER nvparsers
#        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#        PATH_SUFFIXES lib lib64 lib/x64)
#
#find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
#        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#        PATH_SUFFIXES lib lib64 lib/x64)
#
##find_library(TRT_ONNXPARSER_RUNTIME libnvonnxparser_runtime.so
##        HINTS  /usr/lib/aarch64-linux-gnu/)
#
##find_library(TRT_ONNXPARSER_RUNTIME libnvonnxparser_runtime.so
##        HINTS  /home/stephen/Desktop/workplace/TensorRT-CenterNet-master)
#
#find_library(TRT_ONNXPARSER_RUNTIME libnvonnxparser_runtime.so
#        HINTS  /home/stephen/Desktop/workplace/CenterNet_TensorRT_CPP_TRT5.0)
#
#set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN}
#        ${TENSORRT_LIBRARY_PARSER} ${TENSORRT_LIBRARY_ONNXPARSER} ${TRT_ONNXPARSER_RUNTIME})
#message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
#
#find_package_handle_standard_args(
#        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
#
#if(NOT TENSORRT_FOUND)
#    message(ERROR
#            "Cannot find TensorRT library.")
#endif()

#end TensorRT






#指出所有的头文件 包含openCV CUDA CUDNN include
#include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/include )
#include_directories(${OpenCV_INCLUDE_DIRS}
#        ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR}
#        ${TENSORRT_INCLUDE_DIR}
#        ${CMAKE_CURRENT_LIST_DIR}/include
#        ${CMAKE_CURRENT_LIST_DIR}/onnx-tensorrt)
#link_directories(${OpenCV_LIB_DIR})



#指出所有源文件
#file(GLOB CPP_SRC src/*.cpp) ## src下所有的cpp文件
#file(GLOB CU_SRC src/*.cu) ## src下cu文件 必须加上



#cuda_add_library(cttrt SHARED ${CPP_SRC} ${CU_SRC})
#target_link_libraries(cttrt
#        ${TENSORRT_LIBRARY_INFER}
#        ${OpenCV_LIBS}
#        nvonnxparser
#        nvonnxparser_runtime)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
#set(src ${src}  example/demo.cpp example/buildEngine.cpp example/webcam_demo.cpp)


#add_executable(main_test ${src})
#target_link_libraries(main ${OpenCV_LIBS} ${TENSORRT_LIBRARY})


#cuda_add_executable(demo example/demo.cpp  ${CPP_SRC} ${CU_SRC})
#target_link_libraries(demo ${OpenCV_LIBS} ${TENSORRT_LIBRARY})
#
#
#cuda_add_executable(buildEngine example/buildEngine.cpp ${CPP_SRC} ${CU_SRC})
#target_link_libraries(buildEngine ${OpenCV_LIBS} ${TENSORRT_LIBRARY})

#add_executable(buildEngine example/buildEngine.cpp)
#target_link_libraries(buildEngine cttrt)


#cuda_add_executable(webcam_demo example/webcam_demo.cpp ${CPP_SRC} ${CU_SRC})
#target_link_libraries(webcam_demo ${OpenCV_LIBS} ${TENSORRT_LIBRARY})
#cuda_add_executable(webcam_demo example/webcam_demo.cpp)
#target_link_libraries(webcam_demo cttrt)



#单独编译onnx-tensorrt
add_subdirectory(onnx-tensorrt)
add_subdirectory(src)
add_subdirectory(example)


#add_executable(demo example/demo.cpp ${src})
#target_link_libraries(demo ${OpenCV_LIBS})