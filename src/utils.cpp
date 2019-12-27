//
// Created by StephenFang on 2019/12/24.
//
#include <utils.h>


cv::Scalar randomColor(cv::RNG& rng) {
    int icolor = (unsigned) rng;
    return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}