<h2 align = center>CenterNet_TensorRT_CPP</h2>

## update

> 2020.2.20 更新 bbox 

> 2020.2.19 更新 可以build engine for DCNv2, webcam_demo 测试OK 速度有点慢 约500ms

## introduction

plan to finish basic TensorRT version of CenterNet working on **JetsonNano**

most of main code from [Cao](https://github.com/CaoWGG/TensorRT-CenterNet)

I change some to run on JetsonNano， there are still many places need to be enhanced.

still in process ….

## Environment

1. Nvidia Jetson Nano with CSI Camera
2. TensorRT 5.1.2.6

## onnx model
Download here [ctdet_coco_dla_2x.onnx](https://pan.baidu.com/s/10K8EU0uIo91wrdhze2xMZA) Baidu 提取码 3ahy

## Installation
1. git clone https://github.com/Stephenfang51/CenterNet_TensorRT_CPP
2. cd to the repo
3. follow below
```
mkdir build
cd build
cmake ..
make
```

## usage
firstly you should build engine from onnx

1. building Engine : `./buildEngine -i /path/to/xxxxxx.onnx - o /path/to/xxxxxx.engine`

2. webcam demo : `./webcam_demo -e /path/to/xxxx.engine -c true`

3. image demo : `./demo -e /path/to/xxxx.engine -i /path/to/xxxx.jpg`

4. video demo : still working on it....


## image demo result
<img src="https://github.com/Stephenfang51/CenterNet_TensorRT_CPP/blob/master/img/test_detection.jpg">
<img src="https://github.com/Stephenfang51/CenterNet_TensorRT_CPP/blob/master/img/test_detection2.jpg">





