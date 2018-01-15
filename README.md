# DeepLearning Object Detection and Depth Estimation

## Introduction

This project creates an output image that combines a depth estimation with a object detection

### Quick Start:

* download http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy and save it in the source directory.
* save your image in \dataset\single_prediction\image_1
* change image pathes in cnn.py
* start cnn.py
* mark and run code line 200 - 300 (depth prediction of the image)
* mark and run code line 299 - 430 (tensorflow object api, will be downloaded if not available)
* mark and run codee line 431 - end (combines depth image with detection image)


## this repository needs the NYU_FCRN pretrained model for depth estimation you can download it here:
http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy 
(from this github page: https://github.com/iro-cp/FCRN-DepthPrediction)
and save it in the source directory.


## Object Classification: For object classification it uses the resnet50 from keras which will automatically be downloaded by starting the python cnn file
## Object Localization: for object localization it uses Tensorflow Object API it automatically downloads the ssd_mobilenet_v1_coco_11_06_2017 model form 
http://download.tensorflow.org/models/object_detection/
github page (https://github.com/tensorflow/models/tree/master/research/object_detection)

## The network creates: 
					 a depth image     yourImage_depth.jpg
					 a box image       yourImage_box.jpg
					 a combined image  yourImage_combined.jpg
					 
in the folder wher yourImage.jpg is


## Authors

* **Simon Preisinger** - *code* - [SimonPreisinger](https://github.com/SimonPreisinger)
* **Michael Krissgau** - *code* - [MichaelKrissgau](https://github.com/SimonPreisinger)
