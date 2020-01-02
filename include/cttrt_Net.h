//
// Created by StephenFang on 2019/12/24.
//

#ifndef CENTERNET_TRT_CTTRT_NET_H
#define CENTERNET_TRT_CTTRT_NET_H

#include <vector>
#include <algorithm>
#include <string>
#include <numeric>
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include <cttrt_Net.h>
#include <utils.h>
#include <cxxopts.hpp>
namespace cttrt {
    enum struct RUN_MODE
    {
        FLOAT32 = 0 ,
        FLOAT16 = 1 ,
        INT8    = 2
    };

    class cttrtNet {

    public:

        cttrtNet(const std::string &onnxfile,
                 const std::string &calibfile = 0,
                 RUN_MODE mode = RUN_MODE::FLOAT32);


        cttrtNet(const std::string &enginefile);

        ~cttrtNet() {
            ///?
        }


//        void saveEngine(const cxxopts::OptionValues & file_path);
        void saveEngine(const std::string & file_path);


        void doInference(const void *inputData, void *outputData, int batchSize = 1);

        //void printTime()??

        int64_t outputBufferSize;

    private:

        void InitEngine();

        //set all null_ptr
        nvinfer1::IExecutionContext *mTrtContext = nullptr;
        nvinfer1::ICudaEngine *mTrtEngine = nullptr;
        nvinfer1::IRuntime *mTrtRunTime = nullptr;

        //    nvinfer1::IExecutionContext* mContext;
        //    nvinfer1::ICudaEngine* mEngine;
        //    nvinfer1::IRuntime* mRunTime;

        RUN_MODE runMode;

        std::vector<void *> mTrtCudaBuffer;
        std::vector<int64_t> mTrtBindBufferSize;

        void *cudaOutputBuffer;
        cudaStream_t mTrtCudaStream;

        int runIters = 0;
        //int runIters;
        //Profiler mProfiler;

    };

} //namespace cttrt closed
#endif //CENTERNET_TRT_CTTRT_NET_H
