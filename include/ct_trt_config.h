//
// Created by StephenFang on 2019/12/24.
//

#ifndef CENTERNET_TRT_CT_TRT_CONFIG_H
#define CENTERNET_TRT_CT_TRT_CONFIG_H

namespace cttrt {

    static float visThresh = 0.3; //?

    static int inputSize = 512 ;
    static int channel = 3 ;
    static int ouputSize = inputSize/4 ;
    static int kernelSize = 4 ;


    //for center Net detection
    static int classNum = 2 ;
    static float mean[]= {0.485,0.456,0.406};
    static float std[] = {0.229,0.224,0.225};
//    static char *className[]= {(char*)"person",(char*)"helmet"};
}

#endif //CENTERNET_TRT_CT_TRT_CONFIG_H
