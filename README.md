# DeepLearning Object Detection and Depth Estimation

## Introduction

### Quick Start:



## this repository needs the NYU_FCRN pretrained model for depth estimation you can download it here:
http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy 
(from this github page: https://github.com/iro-cp/FCRN-DepthPrediction)
and save it in the source directory.

also it needs images for calulating the object location and depth put it in the folder \dataset\single_prediction

for object classification it uses the resnet50 from keras which will automatically be downloaded by starting the python cnn file

for object localization it uses Tensorflow Object API it automatically downloads the ssd_mobilenet_v1_coco_11_06_2017 model form 
http://download.tensorflow.org/models/object_detection/
github page (https://github.com/tensorflow/models/tree/master/research/object_detection)

The network creates: a depth image     yourImage_depth.jpg
					 a box image       yourImage_box.jpg
					 a combined image  yourImage_combined.jpg
					 
in the folder wher yourImage.jpg is


## Authors

* **Simon Preisinger** - *code* - [SimonPreisinger](https://github.com/SimonPreisinger)
* **Michael Krissgau** - *code* - [MichaelKrissgau](https://github.com/SimonPreisinger)
