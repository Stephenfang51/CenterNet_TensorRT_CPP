//
// Created by StephenFang on 2019/12/25.
// Code borrowed from CAO
//

#ifndef CENTERNET_TRT_CT_FORWARD_H
#define CENTERNET_TRT_CT_FORWARD_H

#include <utils.h>

extern "C" void CTdetforward_gpu(const float *hm, const float *reg,const float *wh ,float *output,
                                 const int w,const int h,const int classes,const int kernerl_size,const float visthresh  );
extern "C" void CTfaceforward_gpu(const float *hm, const float *wh,const float *reg,const float* landmarks,float *output,
                                  const int w,const int h,const int classes,const int kernerl_size, const float visthresh );

#endif //CENTERNET_TRT_CT_FORWARD_H
