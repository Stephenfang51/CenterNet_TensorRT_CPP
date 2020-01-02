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


    auto result = parse(argc, argv);


    std::string input_onnxfile = result["input_onnxfile"].as<std::string>();
    std::string output_engine_file = result["output_engine_file"].as<std::string>();



    cttrt::RUN_MODE mode = cttrt::RUN_MODE::FLOAT32;
    if(result["mode"].as<int>() == 0 ) mode = cttrt::RUN_MODE::FLOAT32;
    if(result["mode"].as<int>() == 1 ) mode = cttrt::RUN_MODE::FLOAT16;
    if(result["mode"].as<int>() == 2 ) mode = cttrt::RUN_MODE::INT8;

    //TODO cttrtNet 的calibfile 还没建构好 暂时先用input_onnxfile替代
    cttrt::cttrtNet net = cttrt::cttrtNet(input_onnxfile, output_engine_file, mode);
    net.saveEngine(result["output_engine_file"].as<std::string>());
    std::cout << "save Engine sucessfully" << std::endl;

    return 0;
}