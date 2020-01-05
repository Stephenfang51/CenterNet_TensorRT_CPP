//
// Created by StephenFang on 2019/12/25.
// Code borrowed from CAO
//

#ifndef CENTERNET_TRT_MAX_POOLING_H
#define CENTERNET_TRT_MAX_POOLING_H

#include <utils.h>

extern "C" void maxPooling_gpu(const float *hm, const float *reg,const float *wh ,float *output,
                                 const int w,const int h,const int classes,const int kernerl_size,const float visthresh  );

#endif //CENTERNET_TRT_CT_FORWARD_H
