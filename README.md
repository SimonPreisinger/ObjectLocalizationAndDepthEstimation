# DeepLearning Object Detection and Depth Estimation

## Introduction

This project creates an output image that combines a depth estimation with a object detection

### Quick Start:

* clone repository to C
* download http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip and extract it in the source directory.
* save your image in \dataset\single_prediction\image_1
* change image pathes in cnn.py
* start cnn.py
* mark and run code

## Object Classification: 
For object classification it uses the resnet50 from keras which will automatically be downloaded by starting the python cnn file
## Object Localization: 
For object localization it uses Tensorflow Object API it automatically downloads the ssd_mobilenet_v1_coco_11_06_2017 model form 
http://download.tensorflow.org/models/object_detection/
github page (https://github.com/tensorflow/models/tree/master/research/object_detection)

## The network creates: 
					 a depth image     yourImage_depth.jpg
					 a box image       yourImage_box.jpg
					 a combined image  yourImage_combined.jpg
					 
in the folder where yourImage.jpg is
## Examples
![Alt text](/image_1_combined.jpg?raw=true "Dog")
![Alt text](/image_2_combined.jpg?raw=true "Cat")

## Authors

* **Simon Preisinger** - *code* - [SimonPreisinger](https://github.com/SimonPreisinger)
* **Michael Krissgau** - *code*
