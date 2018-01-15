# -*- coding: utf-8 -*-
#CNN
#Importing the Keras libraries and packages

# Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils.np_utils import to_categorical 
from keras.callbacks import TensorBoard
from IPython.display import SVG
import matplotlib.pyplot as plt
import cv2 
import numpy as np 
import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image



from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
#import pydot
from keras.utils.vis_utils import model_to_dot

from time import time
import keras



from keras import backend as K
K.clear_session()


# Initialising the CNN
classifier = Sequential();
# first layer: Step 1 - Convolution # Step 2 - Pooling
input_size = (128, 128)
classifier.add(Convolution2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu'))
#classifier.add(Convolution2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu')) #64x64 pixel size (increase using GPU), 3 for colord Image
classifier.add(MaxPooling2D(pool_size = (2, 2)))
###########################################################
# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu')) #Tut Step10 Verbesserung, 2. Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))
###########################################################
# Adding a third convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu')) #Tut Step10 Verbesserung, 3. Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
#classifier.add(Flatten(input_shape=train_data.shape[1:]))
classifier.add(Flatten())

# Step 4 - Full connection
#classifier.add(Dense(output_dim = 128, activation = 'relu')) # Zahl nicht zu klein dass es ein gutes Model ist und nicht zu groß, sonst zu rechenaufwändig
#classifier.add(Dropout(0.5))
#classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #sigmoid für binary werte bei mehr Werten softmax verwenden!
# Step 4 - Multi-Class Classification
classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(3))
classifier.add(Activation('softmax'))
#Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics =['accuracy'] ) # more than two(cats dogs birds)  loss = crossentropy!
                            #'adam'
                            
# Fitting the CNN to the images
# https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
####Train Data
test_datagen = ImageDataGenerator(rescale=1./255)
batch_size=32
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=input_size, #has to be input pixel size
        batch_size=batch_size,
        class_mode='categorical') # more than two: use ...
nb_train_samples = len(training_set.filenames)  #number training images
num_classes = len(training_set.class_indices)  #number classes
#predict_size_train = int(math.ceil(nb_train_samples / batch_size)) 
# labels to categorical [0,0,1]..
train_labels = training_set.classes 
train_labels = to_categorical(train_labels, num_classes=num_classes) 
# save the class indices to use use later in predictions
np.save('class_indices.npy', training_set.class_indices)

### Test Data
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical')
nb_validation_samples = len(test_set.filenames) 
num_classes = len(test_set.class_indices) 
# labels to categorical [0,0,1]..
test_labels = test_set.classes 
test_labels = to_categorical(test_labels, num_classes=num_classes) 

# Start Tensorboard:
# C:\Users\Simon\Documents\DeepLearning\deepLearning>tensorboard --logdir=logs/
tensorBoard = TensorBoard(log_dir="logs/{}".format(time()), write_graph=True, write_images=True)
# tensorBoard = TensorBoard(log_dir='./logs2', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

#from keras.models import load_model
#### LOAD
#classifier = load_model('savedModels/classifierEpoch50.h5')
####

# evtl add EarlyStopping keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
classifier.fit_generator(
        training_set,
        callbacks=[tensorBoard],
        steps_per_epoch=80/32, # number of images in training set
        epochs=20,
        validation_data=test_set,
        validation_steps=20/32) # images in test_set

#val_acc = Testsetaccuracy
# to be better: increase imgaesize 128x128 instead 64x64....
#ap = argparse.ArgumentParser()
#args = vars(ap.parse_args())

#classifier.save('savedModels/classifierEpoch90Layer3Dropout128X128.h5')  # creates a HDF5 file 'my_model.h5'
#classifier.save(args["model"])
# del classifier  # deletes the existing model

#SVG(model_to_dot(classifier).create(prog='dot', format='svg'))
#Part 3 - Make a Prediction
IMAGE_PATH = 'dataset/single_prediction/image_1.jpg' 
  
orig = cv2.imread(IMAGE_PATH) 
  
#print("[INFO] loading and preprocessing image...") 
image = load_img(IMAGE_PATH, target_size=(128, 128))   #224 224
image = img_to_array(image) 
  
# important! otherwise the predictions will be '0' 
image = image / 255 
  
image = np.expand_dims(image, axis=0) 
result = classifier.predict(image)
#inputarray = image[np.newaxis,...]
prediction = classifier.predict_proba(image)
# Label
pCat = prediction[0,0]*100
pDog = prediction[0,1]*100
pPlane = prediction[0,2]*100
if pCat > pDog and pCat > pPlane:
    label = "Cat"
    pLabel = pCat
elif pDog > pCat and pDog > pPlane:
    label = "Dog"
    pLabel = pDog
elif pPlane > pDog and pPlane > pCat:
    label = "Plane"
    pLabel = pPlane
pLabel = "%.2f" % pLabel    
pLabel = str(pLabel)

#finalResult = label + pLabel
    
    
#inID = prediction[0]  
#class_dictionary = training_set.class_indices 
#inv_map = {v: k for k, v in class_dictionary.items()} 
#label = inv_map[inID] 
imageLabel = label+" ("+pLabel+"%)"
#print("Image ID: {}, Label: {}".format(inID, label)) 
cv2.putText(orig, "{}".format(imageLabel), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2) 
cv2.imshow("Classification", orig) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

# Part 3 - Making new predictions
#import numpy as np
#from keras.preprocessing import image
#
##test image
#test_image = image.load_img('dataset/single_prediction/cat_or_dog_3_depth.jpg', target_size = (128,128))
#test_image = image.img_to_array(test_image) # Bild zu Bild mit 3 Farbkanälen umwandeln
#test_image = np.expand_dims(test_image, axis = 0)
##Ergebnis:
#result = classifier.predict(test_image)
#training_set.class_indices #welcher Wert passt zu welchem Label(z.B. cat = 0,dog = 1)
#
#if result[1][0] == 1:
#
#    prediction = 'dog'
#
#else:
#
#    prediction = 'cat'

#PREDICTION IN ORIGINAL IMAGE COPY
#load model
    


from keras import backend as K
K.clear_session()


##################### FRCN
import argparse
import os
import numpy as np
import tensorflow as tfPredict
from matplotlib import pyplot as plt
from PIL import Image      
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#Pathes
model_data_path = "C:/ObjectLocalizationAndDepthEstimation/NYU_FCRN.ckpt"
IMAGE_PATH = 'dataset/single_prediction/image_1.jpg' 
IMAGE_PATH_withoutEnd = IMAGE_PATH
IMAGE_PATH_withoutEnd = IMAGE_PATH_withoutEnd[:-4]

def predict(model_data_path, IMAGE_PATH):

    # Orginal Image size
    orgWidth, orgHeight = Image.open(IMAGE_PATH).size
    channels = 3
    batch_size = 1
    height = 228
    width = 304
       
    # Read image
    img = Image.open(IMAGE_PATH)
    img = img.resize([width,height], Image.ANTIALIAS)

    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
       
    # Create a placeholder for the input image
    input_node = tfPredict.placeholder(tfPredict.float32, shape=(None, height, width, channels))
    from models import fcrn
    from models import network
    # Construct the network
    net = fcrn.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
    
 

    with tfPredict.Session() as sess:
    
        
        # Load the converted parameters
        print('Loading the model')
    
        # Use to load from ckpt file
        saver = tfPredict.train.Saver()     
        saver.restore(sess, model_data_path)
    
        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result

        fig = plt.figure()


        fig, ax1= plt.subplots(1)
        
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ii = plt.imshow(pred[0,:,:,0], interpolation='bicubic')
        axins1 = inset_axes(ax1,
                            width="50%",  # width = 10% of parent_bbox width
                            height="5%",  # height : 50%
                            loc=2)
        fig.colorbar(ii, cax=axins1, orientation="horizontal")   

        fig.savefig(IMAGE_PATH_withoutEnd + "_depth.jpg")
    
        plt.close(fig)
       
    return pred
    
predict(model_data_path, IMAGE_PATH)     


########################################################################################
## OBJECT LOCALIZATION SSD_MBILENET

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import matplotlib

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


import label_map_util

import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = ("C:\ObjectLocalizationAndDepthEstimation\object_recognition_detection\data\mscoco_label_map.pbtxt")

NUM_CLASSES = 90

# ## Download Model

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())
	print ('Download complete')
else:
	print ('Model already exists')


# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join('\dataset\single_prediction', 'image_{}.jpg'.format(i)) for i in range(1, 4) ]
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Running the tensorflow session

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)

     # plt.imshow(image_np)
      #plt.axis("off")
      
      fig = plt.imshow(image_np)

      plt.axis('off')
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      

      image_path_withoutEnd = image_path
      image_path_withoutEnd = image_path_withoutEnd[:-4]

      plt.savefig(image_path_withoutEnd+ "_box.jpg",bbox_inches = 'tight')

######################################################################################################################

from keras import backend as K
K.clear_session()


############################################################### Object Detection Resnet50
from matplotlib import pyplot as detectPlt
from PIL import Image    
import keras as detectKeras
import cv2 
import numpy as np
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
resnet = detectKeras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

IMAGE_PATH = 'dataset/single_prediction/image_1.jpg' 
IMAGE_PATH_withoutEnd = IMAGE_PATH
IMAGE_PATH_withoutEnd = IMAGE_PATH_withoutEnd[:-4]

img_path = IMAGE_PATH
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = resnet.predict(x)
print('Predicted:', decode_predictions(preds))
name =  decode_predictions(preds)

######################## Label Image
showName = (name[0][0][1], name[0][0][2])
orig = cv2.imread(IMAGE_PATH_withoutEnd + "_box.jpg") 
cv2.putText(orig, "{}".format(showName), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2) 
depthImg = cv2.imread(IMAGE_PATH_withoutEnd + "_depth.jpg")
print(IMAGE_PATH_withoutEnd + "_depth.jpg")
w, h = orig.shape[:2]

resized_image = cv2.resize(depthImg, (h,w))

 # Add orginal + depthImage = bothImages
vis = np.concatenate((orig, resized_image), axis=1)
cv2.imwrite(IMAGE_PATH_withoutEnd + '_combined.jpg', vis)
cv2.imshow("Depth", vis)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

