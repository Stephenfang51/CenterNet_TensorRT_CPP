//
// Created by StephenFang on 2020/1/3.
//
#include <cttrt_detector.h>
#include <utils.h>
#include <cttrt_config.h>



std::vector<float> prepareImage(cv::Mat & img){
    int channel = cttrt::channel; //default = 3
    int inputSize = cttrt::inputSize;
    float scale = std::min(float(inputSize)/img.cols,float(inputSize)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale); //定义缩放后的尺寸

    cv::Mat img_resized;
    cv::resize(img, img_resized, scaleSize, 0, 0);

    cv::Mat sample_resized =cv::Mat(inputSize, inputSize, CV_8UC3,cv::Scalar(0,0,0));
    cv::Rect rect((inputSize- scaleSize.width)/2, (inputSize-scaleSize.height)/2, scaleSize.width,scaleSize.height);


    cv::Mat imageROI = sample_resized(rect);
    img_resized.copyTo(imageROI);

    cv::Mat img_float;
    imageROI.convertTo(img_float, CV_32FC3, 1./255.); //顺便标准化
    img_float = img_float - cv::Scalar(0.485, 0.456, 0.406); // 减去均值


    std::vector<cv::Mat> tem_input_channels(channel); //3 channels
    cv::split(img_float, tem_input_channels); //split img_float into 3 different channels

    cv::Mat B = tem_input_channels.at(0) / 0.229;
    cv::Mat G = tem_input_channels.at(1) / 0.224;
    cv::Mat R = tem_input_channels.at(2) / 0.225;

    std::vector<cv::Mat> result;
    result.push_back(B);
    result.push_back(G);
    result.push_back(R);

    std::vector<float> result_img;
    cv::merge(result, result_img);

    return result_img;


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


//TODO 编写中，传入检测结果在传入img， 将检测到的label和bbox画在图上
void drawbbox(const std::vector<Detection> & result, cv::Mat & img){
    int mark;
    int box_think = (img.rows+img.cols) * .001 ;
    float label_scale = img.rows * 0.0009;
    int base_line ;
    cv::Scalar color = (200, 140, 200);


    for (const auto & item : result){
        std::string label;
        std::stringstream stream;

        stream << cttrt::className[item.classId] << " " << item.prob << std::endl;
        std::getline(stream, label);

        //get height, width of textbox
        auto size = cv::getTextSize(label, cv::FONT_HERSHEY_COMPLEX, label_scale,  1, &base_line);

        cv::rectangle(img, cv::Point(item.bbox.x1,item.bbox.y1),
                      cv::Point(item.bbox.x2 ,item.bbox.y2),
                      color[item.classId], box_think, 8, 0);

        cv::putText(img, label, cv::Point(item.bbox.x2, item.bbox.y2 - size.height), cv::FONT_HERSHEY_COMPLEX, label_scale, color, box_think/3, 8, 0);
    }
}//drawbox closed