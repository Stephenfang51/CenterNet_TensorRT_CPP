//
// Created by StephenFang on 2020/1/3.
//

#ifndef CENTERNET_TRT_CTTER_DETECTOR_H
#define CENTERNET_TRT_CTTER_DETECTOR_H

#include <utils.h>

//namespace cttrt {
//    std::vector<float> prepareImage(cv::Mat & img);
//    void post_process(std::vector<Detection> & result, const cv::Mat& img)
//
//} //namespace cttrt closed

extern std::vector<float> prepareImage(cv::Mat & img);
extern void post_process(std::vector<Detection> & result, const cv::Mat& img);
extern void drawbbox(const std::vector<Detection> & result, cv::Mat & img);


#endif //CENTERNET_TRT_CTTER_DETECTOR_H
