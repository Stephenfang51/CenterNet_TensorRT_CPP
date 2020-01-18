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


    //for COCO detection
    static int classNum = 80 ;
    constexpr static char *className[] = {
            (char*)"person", (char*)"bicycle", (char*)"car", (char*)"motorcycle", (char*)"airplane",
            (char*)"bus", (char*)"train", (char*)"truck", (char*)"boat", (char*)"traffic light", (char*)"fire hydrant",
            (char*)"stop sign", (char*)"parking meter", (char*)"bench", (char*)"bird", (char*)"cat", (char*)"dog", (char*)"horse",
            (char*)"sheep", (char*)"cow", (char*)"elephant", (char*)"bear", (char*)"zebra", (char*)"giraffe", (char*)"backpack",
            (char*)"umbrella", (char*)"handbag", (char*)"tie", (char*)"suitcase", (char*)"frisbee", (char*)"skis",
            (char*)"snowboard", (char*)"sports ball", (char*)"kite", (char*)"baseball bat", (char*)"baseball glove",
            (char*)"skateboard", (char*)"surfboard", (char*)"tennis racket", (char*)"bottle", (char*)"wine glass",
            (char*)"cup", (char*)"fork", (char*)"knife", (char*)"spoon", (char*)"bowl", (char*)"banana", (char*)"apple", (char*)"sandwich",
            (char*)"orange", (char*)"broccoli", (char*)"carrot", (char*)"hot dog", (char*)"pizza", (char*)"donut", (char*)"cake",
            (char*)"chair", (char*)"couch", (char*)"potted plant", (char*)"bed", (char*)"dining table", (char*)"toilet", (char*)"tv",
            (char*)"laptop", (char*)"mouse", (char*)"remote", (char*)"keyboard", (char*)"cell phone", (char*)"microwave",
            (char*)"oven", (char*)"toaster", (char*)"sink", (char*)"refrigerator", (char*)"book", (char*)"clock", (char*)"vase",
            (char*)"scissors", (char*)"teddy bear", (char*)"hair drier", (char*)"toothbrush"
    };
 
//    static char *className[]= {(char*)"person",(char*)"helmet"};
}

#endif //CENTERNET_TRT_CT_TRT_CONFIG_H
