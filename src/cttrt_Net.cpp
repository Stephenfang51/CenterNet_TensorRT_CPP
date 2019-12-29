//
// Created by StephenFang on 2019/12/26.
//

#include <cttrt_Net.h>
#include <cttrt_forward.h>
#include <assert.h>
#include <fstream>

//#include <entroyCalibrator.h>

static Logger gLogger;

namespace cttrt
{
//    cttrtNet::cttrtNet(const std::string & onnxfile,
//                        const std::string& calibfile,
//                        RUN_MODE mode): mContext(nullptr), mEngine(nullptr),
//                        mRunTime(nullptr), runMode(mode), runIters(0)
    cttrtNet::cttrtNet(const std::string & onnxfile,
                       const std::string& calibfile,
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


        mRunTime = nvinfer1::createInferRuntime(gLogger);
        mEngine= mRunTime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size(), nullptr);
        //执行反序列因为要开始inference

        serializedEngine->destroy(); //反序列结束可以destroy了



    } //first 构造函数


    cttrtNet::cttrtNet(const std::string&enginefile){
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

        std::cout << "deserializing " << std::endl;


        mRunTime = nvinfer1::createInferRuntime(gLogger);
        mEngine = mRunTime->deserializeCudaEngine(data.get(), length, nullptr);
        //(const void * 	blob, std::size_t 	size, IPluginFactory * 	pluginFactory)

//        InitEngine();

    }//constructor （engine）




    //preprocess for inference
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
//            mBindBufferSizes[i] = totalSize;
//            mCudaBuffers[i] = safeCudaMalloc(totalSize);
//        }
//
//
//
//
//
//
//    }

    void cttrtNet::saveEngine(const std::string & file_path){
        if(mEngine){
            nvinfer1::IHostMemory* serialized_model  = mEngine->serialize(); //将反序列的 再次序列回去
            std::ofstream file(file_path, std::ios::binary | std::ios::out);
            file.write((char*)(serialized_model -> data()), serialized_model -> size());
            file.close();

        }
    }
} //namespace cttrt closed


/////下面要删除
//std::ofstream ofs("serialized_engine.trt", std::ios::out | std::ios::binary);
//ofs.write((char*)(serialized_model ->data()), serialized_model ->size());
//ofs.close();
//
//
//if(mEngine)
//{
//nvinfer1::IHostMemory* data = mEngine->serialize(); //创建data指针 指向 engine 序列化
//std::ofstream file; //ofstream 读取文件
//file.open(fileName,std::ios::binary | std::ios::out); //为输出而打开文件（以二进制的方式）
//
//if(!file.is_open())//如果没有成功开启
//{
//std::cout << "read create engine file" << fileName <<" failed" << std::endl;
//return;
//}
//
//file.write((const char*)data->data(), data->size()); //写入engine
//file.close();
//}