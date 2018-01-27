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
from keras.utils.vis_utils import model_to_dot
from time import time
import keras




#################### FRCN DEPTH ###############################################
from keras import backend as K
K.clear_session()
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










######################## OBJECT LOCALIZATION SSD_MOBILENET######################
from keras import backend as K
K.clear_session()
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
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = ("C:\ObjectLocalizationAndDepthEstimation\object_recognition_detection\data\mscoco_label_map.pbtxt")

NUM_CLASSES = 90

### Download Model

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

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

##################### Change Range for more than one picture
# # Detection
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join('C:\ObjectLocalizationAndDepthEstimation\dataset\single_prediction', 'image_{}.jpg'.format(i)) for i in range(1, 2) ]
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
      
      fig = plt.imshow(image_np)

      plt.axis('off')
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      

      image_path_withoutEnd = image_path
      image_path_withoutEnd = image_path_withoutEnd[:-4]

      plt.savefig(image_path_withoutEnd+ "_box.jpg",bbox_inches = 'tight')








################# Object ClassiFication Resnet50 ###################################
from keras import backend as K
K.clear_session()
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

# Label Image
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

