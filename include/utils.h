//
// Created by StephenFang on 2019/12/24.
//

#ifndef CENTERNET_TRT_UTILS_H
#define CENTERNET_TRT_UTILS_H

#include <map>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cublas_v2.h>
#include <cudnn.h>
#include <assert.h>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <vector>



#ifndef BLOCK
#define BLOCK 512
#endif
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#endif





class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
            : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
    Severity reportableSeverity;
}; //closed Logger



///带入trt的类 dim， 主要获取数据的长度
inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
};

///获取trt的DataType 类型的size
inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
} //close getElementSize


///将cudaMalloc重新封装， 添加了避免显存不够的问题
inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    //&deviceMem为指针的地址
    //memSize 为自己定义要分配多大的尺寸

    if (deviceMem == nullptr) //如果分配结束的deviceMme为空指针， 表示显存不够分配
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
} //closed safeCudaMalloc


struct Box{
    float x1;
    float y1;
    float x2;
    float y2;
};
struct landmarks{
    float x;
    float y;
};
struct Detection{
    //x1 y1 x2 y2
    Box bbox;
    //float objectness;
    int classId;
    float prob;
    landmarks marks[5];
};




extern cv::Scalar randomColor(cv::RNG& rng);
extern dim3 cudaGridSize(uint n);


#endif //CENTERNET_TRT_UTILS_H
