//
// Created by StephenFang on 2020/1/9.
//

#include <opencv2/opencv.hpp>
#include <ctime>
#include <cxxopts.hpp>
#include <cttrt_detector.h>
#include <cttrt_Net.h>

std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True ";
}

cxxopts::ParseResult
parse(int argc, char* argv[])
{
    try {
        cxxopts::Options options(argv[0],  "using tersorrt to speed up detection");
        options.add_options("group")
                ("e, engine", "the path of engine file", cxxopts::value<std::string>())
                ("c, csi", "usage of CSI_camera", cxxopts::value<bool>()->default_value("false"))
                ;
        auto result = options.parse(argc, argv);
        if (result.count("engine")){

            std::cout << result["engine"].as<std::string>() << std::endl;
        }
        else {
            std::cout << "No engine file provided" << std::endl;
            exit(-1);
        }
        if (result.count("csi")){

            std::cout << "The CSI camera mode set is " << result["csi"].as<bool>() << std::endl;
        }
        else {
            std::cout << "The CS camera mode set is false" << std::endl;
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
    ///判断是否启用CSI Camera


    ///1. get Net done
    std::string engine = result["engine"].as<std::string>();
    cttrt::cttrtNet net(engine, true);
    std::unique_ptr<float[]> outputData(new float[net.outputBufferSize]);

    int capture_width = 1280 ;
    int capture_height = 720 ;
    int display_width = 1280 ;
    int display_height = 720 ;
    int framerate = 30 ;
    int flip_method = 0 ;

    std::string pipeline = gstreamer_pipeline(capture_width,
                                              capture_height,
                                              display_width,
                                              display_height,
                                              framerate,
                                              flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
        std::cout<<"Failed to open camera."<<std::endl;
        return (-1);
    }
    ///如果CSI为true
    if (result["csi"].as<bool>() == true) {





        cv::namedWindow("result", cv::WINDOW_NORMAL);
        cv::resizeWindow("result",1024,768);
        std::cout << "Hit ESC to exit" << "\n" ;


        ///FPS counter
        long frameCounter = 0;
        std::time_t timeBegin = std::time(0);
        int tick = 0;
        int fps = 0;
        cv::Mat frame; //for read
//        cv::Mat show; // for imshow
        //TODO 暂时先不保存video, 看能否提速
//        cv::Mat output; // for writing to disk




        ///define writer
        //TODO 暂时先不保存video, 看能否提速
//        cv::VideoWriter writer;
//        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
//        writer.open("appsrc ! autovideoconvert ! omxh264enc ! matroskamux ! filesink location=test.mp4 ", codec, (double)25, cv::Size(1280, 720), true);
//
//        if (!writer.isOpened()) {
//            std::cout << "Could not open the output video file for write\n";
//            return -1;
//        }

//        while(true)
        while(true)
        {

            if (!cap.read(frame))
            {
                std::cout<<"\n Cannot read the video file. \n";
                break;
            }
//            cap.read(frame);
//            show = frame;
            auto inputData = prepareImage(frame);
            net.doInference(inputData.data(), outputData.get());

            int num_det = static_cast<int>(outputData[0]); //num of detection
            std::vector<Detection> det_result;
            det_result.resize(num_det);
            memcpy(det_result.data(), &outputData[1], num_det * sizeof(Detection));

            post_process(det_result, frame);

            drawbbox(det_result, frame); //draw bbox on frame
            //write to disk after drawbbox
            ///count fps
            frameCounter++;
            std::time_t timeNow = std::time(0) - timeBegin;
            if (timeNow - tick >= 1)
            {
                tick++;
                fps = frameCounter;
                frameCounter = 0;
            }

            ///put fps text
//            cv::putText(frame, cv::format("Average FPS=%d",fps),
//                        cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255));

            //TODO 暂时先不保存video, 看能否提速
//            frame.convertTo(output, CV_8UC3);
//            writer.write(output); //write

//            cv::imshow("result", frame);
//            int keycode = cv::waitKey(30) & 0xff ;
            int keycode = cv::waitKey(30) & 0xff ;
            if (keycode == 27) break ;
            }
        cap.release();
        cv::destroyAllWindows() ;
        return 0;
        }

    }
        ///主要for non-gstreamer
//    else {
//        cv::VideoCapture cap(0);
//        cv::VideoWriter writer;
//        writer.open("output.mp4", CV_FOURCC('M', 'J', 'P', 'G'), (double)25, cv::Size(640, 480));
//        cv::Mat frame;
//        cv::Mat output;
//        while(true)
//        {
//            cap.read(frame);
//            if (!cap.read(frame)) // if not success, break loop
//                // read() decodes and captures the next frame.
//            {
//                std::cout<<"\n Cannot read the video file. \n";
//                break;
//            }
//
//            auto inputData = prepareImage(frame);
//            net.doInference(inputData.data(), outputData.get());
//
//            int num_det = static_cast<int>(outputData[0]); //num of detection
//            std::vector<Detection> result;
//            result.resize(num_det);
//            memcpy(result.data(), &outputData[1], num_det * sizeof(Detection));
//
//            post_process(result, frame);
//
//
//            drawbbox(result, frame);
//            frame.convertTo(output, CV_8UC3);
//            writer.write(output);
//
//            cv::imshow("det result", frame);
//            cv::waitKey(0);
//        }
//        cap.release();
//        cv::destroyAllWindows() ;
//        return 0;
//    }



//} //closed