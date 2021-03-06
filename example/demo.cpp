//
// Created by StephenFang on 2019/12/24.
//


#include <cxxopts.hpp>
#include <string>
#include <iostream>
#include <cttrt_config.h>
#include <cttrt_Net.h>
#include <utils.h>
#include <cttrt_detector.h>
#include <memory>


//cxxopts::Options options("MyProgram", "One line description of MyProgram");
//options.add_options()
//("f,file", "File name", cxxopts::value<std::string>());
cxxopts::ParseResult
parse(int argc, char* argv[])
{
    try {
        cxxopts::Options options(argv[0],  "using tersorrt to speed up detection");
        options.add_options("groups")
                ("e, engine", "the path of engine file", cxxopts::value<std::string>())
                ("i, image", "the path of test image", cxxopts::value<std::string>())
                ("v, video", "the path of test video", cxxopts::value<std::string>())
                ;
        auto result = options.parse(argc, argv);
        if (result.count("engine")){

            std::cout << result["engine"].as<std::string>() << std::endl;
        }
        else {
            std::cout << "No engine file provided" << std::endl;
            exit(-1);
        }
        if (result.count("image")){
            std::cout << result["image"].as<std::string>() << std::endl;
        }
        else {
            std::cout << "No test image provided" << std::endl;
        }
        if (result.count("video")){
            std::cout << result["video"].as<std::string>() << std::endl;
        }
        else {
            std::cout << "No test video provided" << std::endl;
        }

        if (!result.count("image")  && !result.count("video")){
            std::cout << "Please input any image or video" << std::endl;
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

//    cv::RNG rng(244);
//
//    std::vector<cv::Scalar> color;
//    for (int i=0; i < cttrt::classNum; i++)
//    {
//        color.push_back(randomColor(rng));
//    }

    auto result = parse(argc, argv);

//    cv::namedWindow("detection", cv::WINDOW_NORMAL);
//    cv::resizeWindow("detection", 1024, 768);

    std::string engine = result["engine"].as<std::string>();


    ///1. get Net done
    cttrt::cttrtNet net(engine, true); //TODO true只是区别构造函数
    std::unique_ptr<float[]> outputData(new float[net.outputBufferSize]);



    if (result.count("image")){
        std::string image = result["image"].as<std::string>();
        cv::Mat img = cv::imread(image);
        auto inputData = prepareImage(img);

        net.doInference(inputData.data(), outputData.get());

        int num_det = static_cast<int>(outputData[0]); //num of detection
        std::vector<Detection> result;
        result.resize(num_det);
        memcpy(result.data(), &outputData[1], num_det * sizeof(Detection));

        post_process(result, img);
        drawbbox(result, img);

//        cv::imshow("det result", img);
        cv::imwrite("test_detection.jpg", img);
        cv::waitKey(0);

    }
    //TODO video还没搞定
    if (result.count("video")){
        std::string video = result["video"].as<std::string>();
        cv::Mat frame;
        cv::Mat output;
        cv::VideoCapture cap(video);
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        cv::VideoWriter writer("VideoTest.mp4", codec, 25.0, cv::Size(640, 480));
//        while (cap.read(video))
        while(true)
        {

            cap.read(frame);
            if (!cap.read(frame)) // if not success, break loop
                // read() decodes and captures the next frame.
            {
                std::cout<<"\n Cannot read the video file. \n";
                break;
            }

            auto inputData = prepareImage(frame);
            net.doInference(inputData.data(), outputData.get());

            int num_det = static_cast<int>(outputData[0]); //num of detection
            std::vector<Detection> result;
            result.resize(num_det);
            memcpy(result.data(), &outputData[1], num_det * sizeof(Detection));

            post_process(result, frame);

            drawbbox(result, frame);
//            cv::imshow("det result", frame);
            frame.convertTo(output, CV_8UC3);
            writer.write(output);
            int keycode = cv::waitKey(30) & 0xff ;
            if (keycode == 27) break ;
//            cv::waitKey(0);
        }
        cap.release();
        cv::destroyAllWindows() ;
        return 0;
    }




} // main closed