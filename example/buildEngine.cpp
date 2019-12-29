//
// Created by StephenFang on 2019/12/28.
//

#include <cttrt_Net.h>
#include <cxxopts.hpp>
#include <string>
#include <utils.h>
#include <iostream>

cxxopts::ParseResult
parse(int argc, char* argv[])
{
    try {
        cxxopts::Options options(argv[0], "buildEngine");
        options.add_options("groups")
                ("i, input_onnxfile", "the path of  input-onnxfile (str)", cxxopts::value<std::string>())
                ("o, output_engine_file", "the path of output-enginefile )(str)", cxxopts::value<std::string>())
                ("m, mode", "run-mode (int)", cxxopts::value<int>()->default_value("0"))
                ;
        auto result = options.parse(argc, argv);
        if (result.count("input_onnxfile")){

            std::cout << result["input_onnxfile"].as<std::string>() << std::endl;
        }
        else {
            std::cout << "No input onnx_file" << std::endl;
            exit(-1);
        }
        if (result.count("output_engine_file")){
            std::cout << result["output_engine_file"].as<std::string>() << std::endl;

        }
        else {
            std::cout << "No output engine file" << std::endl;
            exit(-1);
        }

        return result;

    }//try closed
    catch (const cxxopts::OptionException& e)
    {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
}//parse closed

int main(int argc, char* argv[]){
//    cxxopts::Options options("buildEngine", "buildEngine" );
//    options.add_options()
//            ("i, input_onnxfile", "the path of  input-onnxfile (str)", cxxopts::value<std::string>())
//            ("o, output_engine_file", "the path of output-enginefile )(str)", cxxopts::value<std::string>())
//            ("m, mode", "run-mode (int)", cxxopts::value<int>()->default_value("0"))
//            ;
//    auto result = options.parse(argc, argv);
//    if (result.count("input_onnxfile")){
//        std::string input_onnxfile = result["input_onxnfile"].as<std::string>();
//    }
//    else {
//        std::cout << "No input onnx_file" << std::endl;
//        exit(-1);
//    }
//    if (result.count("output_engine_file")){
//        std::string output_engine_file = result["output_engine_file"].as<std::string>();
//    }
//    else {
//        std::cout << "No output engine file" << std::endl;
//        exit(-1);
//    }


    ///以下是测试
    auto result = parse(argc, argv);

    ///以上是测试

    std::string input_onnxfile = result["input_onnxfile"].as<std::string>();
    std::string output_engine_file = result["output_engine_file"].as<std::string>();



    cttrt::RUN_MODE mode = cttrt::RUN_MODE::FLOAT32;
    if(result["mode"].as<int>() == 0 ) mode = cttrt::RUN_MODE::FLOAT32;
    if(result["mode"].as<int>() == 1 ) mode = cttrt::RUN_MODE::FLOAT16;
    if(result["mode"].as<int>() == 2 ) mode = cttrt::RUN_MODE::INT8;

//    cttrt::cttrtNet net = cttrt::cttrtNet(result["input_onnxfile"].as<std::string>(), r.as<std::string>(), mode);
//    cttrt::cttrtNet net = cttrt::cttrtNet(result["input_onnxfile"].as<std::string>(), result["input_onnxfile"].as<std::string, mode);
    cttrt::cttrtNet net = cttrt::cttrtNet(input_onnxfile, input_onnxfile, mode);
    net.saveEngine(result["output_engine_file"].as<std::string>());
    std::cout << "save Engine sucessfully" << std::endl;

    return 0;
}