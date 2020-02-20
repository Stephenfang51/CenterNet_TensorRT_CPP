//
// Created by StephenFang on 2020/1/3.
//
#include <cttrt_detector.h>
#include <utils.h>
#include <cttrt_config.h>



std::vector<float> prepareImage(cv::Mat & img){
//    int channel = cttrt::channel; //default = 3
//    int inputSize = cttrt::inputSize;
//
//    float scale = std::min(float(inputSize)/img.cols,float(inputSize)/img.rows);
//
//    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale); //定义缩放后的尺寸
//
//    cv::Mat img_resized;
//
//    cv::resize(img, img_resized, scaleSize, 0, 0);
//
//
//
//    cv::Mat sample_resized =cv::Mat(inputSize, inputSize, CV_8UC3,cv::Scalar(0,0,0));
//
//    cv::Rect rect((inputSize- scaleSize.width)/2, (inputSize-scaleSize.height)/2, scaleSize.width,scaleSize.height);
//
//
//    cv::Mat imageROI = sample_resized(rect);
//
//    img_resized.copyTo(imageROI);
//
//    cv::Mat img_float;
//
//    imageROI.convertTo(img_float, CV_32FC3, 1./255.); //顺便标准化
//
//    img_float = img_float - cv::Scalar(0.485, 0.456, 0.406); // 减去均值
//
//
//    std::vector<cv::Mat> tem_input_channels(channel); //3 channels
//
//    cv::split(img_float, tem_input_channels); //split img_float into 3 different channels
//
//    cv::Mat B = tem_input_channels.at(0) / 0.229;
//    cv::Mat G = tem_input_channels.at(1) / 0.224;
//    cv::Mat R = tem_input_channels.at(2) / 0.225;
//
//    std::vector<cv::Mat> result;
//
//    result.push_back(B);
//    result.push_back(G);
//    result.push_back(R);
//
//    std::vector<float> result_img;
//
//    cv::merge(result, result_img);
//    std::cout << "19" << std::endl;
//    return result_img;

    int channel = 3;
    int input_w = 512;
    int input_h = 512;
    float scale = cv::min(float(input_w)/img.cols,float(input_h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat resized;
    cv::resize(img, resized,scaleSize,0,0);


    cv::Mat cropped = cv::Mat::zeros(input_h,input_w,CV_8UC3);
    cv::Rect rect((input_w- scaleSize.width)/2, (input_h-scaleSize.height)/2, scaleSize.width,scaleSize.height);

    resized.copyTo(cropped(rect));


    cv::Mat img_float;

    cropped.convertTo(img_float, CV_32FC3,1./255.);

    //HWC TO CHW
    std::vector<cv::Mat> input_channels(channel);
    cv::split(img_float, input_channels);

    // normalize
    std::vector<float> result(input_h*input_w*channel);
    auto data = result.data();
    int channelLength = input_h * input_w;
    static float mean[]= {0.485,0.456,0.406};
    static float std[] = {0.229,0.224,0.225};
    for (int i = 0; i < channel; ++i) {
        cv::Mat normed_channel = (input_channels[i]-mean[i])/std[i];
        memcpy(data,normed_channel.data,channelLength*sizeof(float));
        data += channelLength;
    }
    return result;

} //preprocess closed

void post_process(std::vector<Detection> & result, const cv::Mat& img){
    int inputSize = cttrt::inputSize;
    float scale = std::min(float(inputSize)/img.cols, float(inputSize)/img.rows);


    float dx = (inputSize - scale * img.cols) / 2;
    float dy = (inputSize - scale * img.rows) / 2;

    for (auto & item :result)
    {
        float x1 = (item.bbox.x1 - dx) / scale ;
        float y1 = (item.bbox.y1 - dy) / scale ;
        float x2 = (item.bbox.x2 - dx) / scale ;
        float y2 = (item.bbox.y2 - dy) / scale ;

        //x1, y1 为bbox的左上or左下坐标 不得为负数
        x1 = (x1 > 0 ) ? x1 : 0 ;
        y1 = (y1 > 0 ) ? y1 : 0 ;

        //x2, y2为右下or右上坐标， 不得超过图像边
        x2 = (x2 < img.cols  ) ? x2 : img.cols - 1 ;
        y2 = (y2 < img.rows ) ? y2  : img.rows - 1 ;

        //将修正后的坐标点重新赋值
        item.bbox.x1  = x1 ;
        item.bbox.y1  = y1 ;
        item.bbox.x2  = x2 ;
        item.bbox.y2  = y2 ;
    }

}//postprocess closed


void drawbbox(const std::vector<Detection> & result, cv::Mat & img){
//    int box_think = (img.rows+img.cols) * .001 ;
//    float label_scale = img.rows * 0.0009;
    int base_line ;
    cv::Scalar color = (100, 140, 100);
    cv::Scalar color_text = (255, 255, 255);


    for (const auto & item : result){
//        std::stringstream stream;
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC3);
        //get label text



        //drawing bounding bbox
        cv::rectangle(img, cv::Point(item.bbox.x1,item.bbox.y1),
                      cv::Point(item.bbox.x2 ,item.bbox.y2),
                      color, 2);
        cv::rectangle(mask,cv::Point(item.bbox.x1,item.bbox.y1),
                      cv::Point(item.bbox.x2 ,item.bbox.y2),
                      color, cv::FILLED, 0);
        //creating label
        char score_str[128];
        sprintf(score_str, "%.2f", item.prob);
        std::string label = std::string(cttrt::className[item.classId]) + " " + std::string(score_str);
        //mask for text
        cv::Point text_origin = cv::Point(item.bbox.x1 - 2, item.bbox.y1 - 3);
        //get height, width of label box
        auto text_size = cv::getTextSize(label, cv::FONT_HERSHEY_COMPLEX, 0.6,  2, &base_line);

        //text background
        cv::rectangle(img, cv::Point(text_origin.x, text_origin.y + 2),
                      cv::Point(text_origin.x + text_size.width,
                                text_origin.y - text_size.height),
                      color, -1, 0);


        //put text
        cv::Point text_cordi = cv::Point(item.bbox.x1, item.bbox.y1-5);
        cv::putText(img, label, text_cordi, cv::FONT_HERSHEY_COMPLEX, 0.6, color_text, 2);

    }
    cv::imshow("result", img);
    //TODO decide whether to remove
}//drawbox closed