//
// Created by StephenFang on 2019/12/24.
//


//#include <argparse.h>
#include <cxxopts.hpp>
#include <string>
#include <iostream>
#include <ct_trt_config.h>
//#include <ctdetNet.h>
#include <utils.h>
#include <memory>

//cxxopts::Options options("MyProgram", "One line description of MyProgram");
//options.add_options()
//("f,file", "File name", cxxopts::value<std::string>());
int main(){
    cxxopts::Options options("centerNet_trt", "using tersorrt to speed up detection" );
    options.add_options()
            ("e, engine", "input Engine", cxxopts::value<std::string>())
            ("i, img", "input-demo-img", cxxopts::value<std::string>())
            ("v video", "input-demo-video", cxxopts::value<std::string>())
            ;
    cv::RNG rng(244);

    std::vector<cv::Scalar> color;
    for (int i=0; i < cttrt::classNum; i++)
    {
        color.push_back(randomColor(rng));
    }

    cv::namedWindow("detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("detection", 1024, 768);





}