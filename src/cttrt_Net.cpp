//
// Created by StephenFang on 2019/12/26.
//

#include <cttrt_Net.h>
#include <maxpooling_gpu.h>
#include <assert.h>
#include <fstream>
#include <chrono>
#include <cttrt_config.h>

//#include <entroyCalibrator.h>

static Logger gLogger;

namespace cttrt
{
//
//    cttrtNet::cttrtNet(std::string & onnxfile,
//                       const std::string& calibfile,
//                       RUN_MODE mode): runMode(mode)
    cttrtNet::cttrtNet(std::string & onnxfile,
                       RUN_MODE mode): runMode(mode)
    {
        const int maxBatchSize = 1;
        nvinfer1::IBuilder * builder = nvinfer1::createInferBuilder(gLogger);
        nvinfer1::INetworkDefinition* network = builder->createNetwork();
        //Network empty now

        //TRT7
        //nvinfer1::INetwrokDefinition* network = builder->createNetworkV2(explicitBatch);
        //Note: In TensorRT 7.0, the ONNX parser only supports full-dimensions mode,
        // meaning that your network definition must be created with the explicitBatch flag set. For more information, see Working With Dynamic Shapes.


        //onnxParser 无法编译？？？
//        nvonnxparser::IONNXParser * parser = nvonxparser::createONNXParser(*network, gLogger);
        auto parser = nvonnxparser::createParser(*network, gLogger);
        std::cout << "start parsing model " << std::endl;
        //Ingest the model
//        parser->parseFromFile(onnxfile.c_str(), nvinfer1::ILogger::Severity::kWARNING);
        int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
        parser->parseFromFile(onnxfile.c_str(), verbosity);


        //builder config
        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(1 << 30);
        builder ->setFp16Mode(true);

        //build Engine
        std::cout << "start building engine" << std::endl;
        nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
//        nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network);
        //buildcudaEngine will be removed future
        std::cout << "build engine finished" << std::endl;

        parser->destroy();
        network->destroy();
//        config->destroy();
        builder->destroy();

        nvinfer1::IHostMemory * serializedEngine = engine->serialize();
        ////save to disk ? 原作者这里没有直接save engine to disk, 而是另外做了一个save engine function
        ////covnersion and run Engine should be on the same GPU
        engine->destroy();


        mTrtRunTime = nvinfer1::createInferRuntime(gLogger);
        mTrtEngine= mTrtRunTime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size(), nullptr);
        //执行反序列因为要开始inference

        serializedEngine->destroy(); //反序列结束可以destroy了



    } //first 构造函数

    //TODO 需要增加参数区别构造函数
    cttrtNet::cttrtNet(const std::string&enginefile, bool demo){

        using namespace std;
        std::fstream file;
        file.open(enginefile, ios::binary | ios::in); //二进制方式读取文件
        if (!file.is_open())
        {
            std::cout << "read engine file" << enginefile << "failed" << std::endl;
            return;
        }

        file.seekg(0, ios::end); //从末尾往第0个字节
        int length = file.tellg(); //tellg表示“內置指標”的目前位置, 得出长度
        file.seekg(0, ios::beg);

        std::unique_ptr<char[]> data(new char[length]); //创建一个指针类型data
        file.read(data.get(), length);
        //read ( char * buffer, streamsize size )

        file.close();
        ///反序列化


        mTrtPlugins = nvonnxparser::createPluginFactory(gLogger);
        std::cout << "deserializing " << std::endl;


        mTrtRunTime = nvinfer1::createInferRuntime(gLogger);
        mTrtEngine = mTrtRunTime->deserializeCudaEngine(data.get(), length, mTrtPlugins);
        //(const void * 	blob, std::size_t 	size, IPluginFactory * 	pluginFactory)

//        InitEngine();

    }//constructor （engine）




    //preprocess for inference 以下暂时摆着
//    void cttrt::InitEngine(){
//        nvinfer1::IExecutionContext *context = engine->createExecutionContext();
//
//        int nbBindings = mEngine->getNbBindings(); //get total index num
//
//        assert(nbBindings == 4); //input + output should == 4
//        mCudaBUffers.resize(nbBindings);
//        mBindBufferSizes.resize(nbBindings);
////        int64_t totalSize = 0;
//
//        for (int i =0; i < nbBindings; i++){
//            nvinfer1::Dims dimms = mEngine->getBindingDimensions(i);
//            nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
//
//            int64_t totalSize = volume(dims) * maxBatchSize * getElementSize(dtype);
//
//            mTrtBindBufferSizes[i] = totalSize;
//            mTrtCudaBuffers[i] = safeCudaMalloc(totalSize);
//        }


    void cttrtNet::InitEngine()
        {
            //mTrtBatchSize = mTrtEngine->getMaxBatchSize();
            int maxBatchSize = 1;
            mTrtContext = mTrtEngine->createExecutionContext();
            assert(mTrtContext != nullptr);
//            mTrtContext->setProfiler(&mProfiler);

            // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings()
            int nbBindings = mTrtEngine->getNbBindings();

            mTrtCudaBuffer.resize(nbBindings);
            mTrtBindBufferSize.resize(nbBindings);
            for (int i = 0; i < nbBindings; ++i)
            {
                nvinfer1::Dims dims = mTrtEngine->getBindingDimensions(i);
                nvinfer1::DataType dtype = mTrtEngine->getBindingDataType(i);
                int64_t totalSize = volume(dims) * maxBatchSize * getElementSize(dtype);
                mTrtBindBufferSize[i] = totalSize;
                mTrtCudaBuffer[i] = safeCudaMalloc(totalSize);
//                if(mTrtEngine->bindingIsInput(i))
            }
            outputBufferSize = mTrtBindBufferSize[1] * 6 ;
            cudaOutputBuffer = safeCudaMalloc(outputBufferSize);
            CUDA_CHECK(cudaStreamCreate(&mTrtCudaStream));
        } //InitEngine closed

    void cttrtNet::doInference(const void* inputData, void* outputData ,int batchSize)
    {
//        const int batchSize = 1;

        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        int inputIndex = 0;
        CUDA_CHECK(cudaMemcpyAsync(mTrtCudaBuffer[inputIndex], inputData, mTrtBindBufferSize[inputIndex], cudaMemcpyHostToDevice, mTrtCudaStream));
        auto t_start = std::chrono::high_resolution_clock::now();
        mTrtContext->execute(batchSize, &mTrtCudaBuffer[inputIndex]);

        ///max pooling and post process
        maxPooling_gpu(static_cast<const float *>(mTrtCudaBuffer[1]),static_cast<const float *>(mTrtCudaBuffer[2]),
                         static_cast<const float *>(mTrtCudaBuffer[3]),static_cast<float *>(cudaOutputBuffer),
                         ouputSize,ouputSize,classNum,kernelSize,visThresh);

        auto t_end = std::chrono::high_resolution_clock::now();
        float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();

        std::cout << "Time taken for inference is " << total << " ms." << std::endl;

        CUDA_CHECK(cudaMemcpyAsync(outputData, cudaOutputBuffer, outputBufferSize, cudaMemcpyDeviceToHost, mTrtCudaStream));

        //cudaStreamSynchronize(mTrtCudaStream);

    }


    void cttrtNet::saveEngine(const std::string & file_path){
        if(mTrtEngine){
            nvinfer1::IHostMemory* serialized_model  = mTrtEngine->serialize(); //将反序列的 再次序列回去
            std::ofstream file(file_path, std::ios::binary | std::ios::out);
            file.write((char*)(serialized_model -> data()), serialized_model -> size());
            file.close();

        }
    }
} //namespace cttrt closed


