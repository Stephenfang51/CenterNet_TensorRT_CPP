//
// Created by StephenFang on 2019/12/25.
//

#include <ct_trt_config.h>

__device__ float Logist(float data){ return 1./(1. + exp(-data)); }

///检测前向
//CTdetforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
//static_cast<const float *>(mCudaBuffers[3]),static_cast<float *>(cudaOutputBuffer),
//        ouputSize,ouputSize,classNum,kernelSize,visThresh)
__global__ void CTdetforward_kernel(const float *hm, const float *reg,const float *wh ,
                                    float *output,const int w,const int h,const int classes,const int kernerl_size,const float visthresh  ) {

    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; //当前thread位置
    if (idx >= w*h) return; //?
    int padding = kernerl_size/2; //kernel_size default = 4
    int offset = - padding /2;
    int stride = w*h; //计算跨步
    int grid_x = idx % w ; //取余数
    int grid_y = idx / w ; //取整数
    int cls,l,m;
    float c_x,c_y;
    for (cls = 0; cls < classes; ++cls ) //遍历每一个类别？
    {
        int objIndex = stride * cls + idx; //物件的位置？？
        float objProb = hm[objIndex]; //求出当前类的概率？
        float max=-1;
        int max_index =0;

        for(l=0 ;l < kernerl_size ; ++l) //kernel_size default = 4
            for(m=0 ; m < kernerl_size ; ++m){
                int cur_x = offset + l + grid_x;
                int cur_y = offset + m + grid_y;
                int cur_index = cur_y * w + cur_x + stride*cls;
                int valid = (cur_x>=0 && cur_x < w && cur_y >=0 && cur_y <h );
                float val = (valid !=0 ) ? Logist(hm[cur_index]): -1;
                max_index = (val > max) ? cur_index : max_index;
                max = (val > max ) ?  val: max ;
            }
        objProb = Logist(objProb);
        if((max_index == objIndex) && (objProb > visthresh)){

            int resCount = (int)atomicAdd(output,1);
            //printf("%d",resCount);
            char* data = (char * )output + sizeof(float) + resCount*sizeof(Detection);
            Detection* det =  (Detection*)(data);
            c_x = grid_x + reg[idx] ; c_y  = grid_y + reg[idx+stride];

            ///计算出bbox的坐标
            det->bbox.x1 = (c_x - wh[idx]/2)*4;
            det->bbox.y1 = (c_y - wh[idx+stride]/2)*4 ;
            det->bbox.x2 = (c_x + wh[idx]/2)*4;
            det->bbox.y2 = (c_y + wh[idx+stride]/2)*4;
            det->classId = cls;
            det->prob = objProb;
        }
    }
}

__global__ void CTfaceforward_kernel(const float *hm, const float *wh,const float *reg,const float* landmarks,
                                     float *output,const int w,const int h,const int classes,const int kernerl_size,const float visthresh  ) {
    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= w*h) return;
    int padding = kernerl_size/2;
    int offset = - padding /2;
    int stride = w*h;
    int grid_x = idx % w ;
    int grid_y = idx / w ;
    int cls,l,m,mark_id;
    float c_x,c_y,scale_w,scale_h;
    for (cls = 0; cls < classes; ++cls )
    {
        int objIndex = stride * cls + idx;
        float objProb = hm[objIndex];
        float max=-1;
        int max_index =0;
        for(l=0 ;l < kernerl_size ; ++l)
            for(m=0 ; m < kernerl_size ; ++m){
                int cur_x = offset + l + grid_x;
                int cur_y = offset + m + grid_y;
                int cur_index = cur_y * w + cur_x + stride*cls;
                int valid = (cur_x>=0 && cur_x < w && cur_y >=0 && cur_y <h );
                float val = (valid !=0 ) ? hm[cur_index]: -1;
                max_index = (val > max) ? cur_index : max_index;
                max = (val > max ) ?  val: max ;
            }
        //printf("%f\n",objProb);
        if((max_index == objIndex) && (objProb > visthresh)){

            int resCount = (int)atomicAdd(output,1);
            //printf("%d",resCount);
            char* data = (char * )output + sizeof(float) + resCount*sizeof(Detection);
            Detection* det =  (Detection*)(data);
            c_x = (grid_x + reg[idx+stride] + 0.5)*4 ; c_y  = (grid_y + reg[idx] + 0.5) * 4;
            scale_w =  expf(wh[idx+stride]) * 4 ; scale_h  = expf(wh[idx]) * 4;
            det->bbox.x1 = c_x - scale_w/2;
            det->bbox.y1 = c_y - scale_h/2 ;
            det->bbox.x2 = c_x + scale_w/2;
            det->bbox.y2 = c_y + scale_h/2;
            det->prob = objProb;
            det->classId = cls;
            for(mark_id=0 ; mark_id < 5 ; ++mark_id ){
                det->marks[mark_id].x = det->bbox.x1 + landmarks[idx + (2*mark_id+1)*stride]*scale_w;
                det->marks[mark_id].y = det->bbox.y1 + landmarks[idx + (2*mark_id)*stride]*scale_h;
            }
        }
    }
}

///把前面定义的kernel 交给GPU运行
void CTdetforward_gpu(const float *hm, const float *reg,const float *wh ,float *output,
                      const int w,const int h,const int classes,const int kernerl_size, const float visthresh ){
    uint num = w * h;
    CTdetforward_kernel<<<cudaGridSize(num),BLOCK>>>(hm,reg,wh,output,w,h,classes,kernerl_size,visthresh);
    //执行GPU kernel
}



void CTfaceforward_gpu(const float *hm, const float *wh,const float *reg,const float* landmarks,float *output,
                       const int w,const int h,const int classes,const int kernerl_size, const float visthresh ){
    uint num = w * h;
    CTfaceforward_kernel<<<cudaGridSize(num),BLOCK>>>(hm,wh,reg,landmarks,output,w,h,classes,kernerl_size,visthresh);
}
