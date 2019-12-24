#include <iostream>
#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>

int main(int argc, char* argv[]) {
    cv::Mat test;
    std::cout << "This is a test" << std::endl;



    //cxxopts test
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
            ("f,file", "File name", cxxopts::value<std::string>());
    auto result = options.parse(argc, argv);
    if (result.count("file")){
        std::cout << "OK ! " << std::endl;
    }

    std::cout << result["file"].as<std::string>() << std::endl;
    //cxxopts test end


    return 0;


}