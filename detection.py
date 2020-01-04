import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
from IPython.display import display
import cv2
#import pytesseract

import glob
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from character_recognition import Character_recognition

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def convert_grayscale(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  gray = cv2.medianBlur(gray, 3)
  #img = gray
  return gray

MODEL_NAME = 'trained-inference-graphs'
PATH_TO_CKPT = MODEL_NAME + '/output_inference_graph_v1.pb/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('annotations', 'label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
PATH_TO_RESULT_IMAGES_DIR = 'result'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 29) ]
TEST_IMAGE_PATHS = glob.glob( 'test_images/*.jpg' )
brojac = 1

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      print(image_path)
      image = Image.open(image_path) 
      image_np = load_image_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      ymin = boxes[0,0,0]
      xmin = boxes[0,0,1]
      ymax = boxes[0,0,2]
      xmax = boxes[0,0,3]
      (im_width, im_height) = image.size
      (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
      cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn),int(ymaxx - yminn), int(xmaxx - xminn))
      
      img_data = sess.run(cropped_image)       
      
      im2 = Image.fromarray(img_data)
      img_path = os.path.join(PATH_TO_RESULT_IMAGES_DIR, 'image{}.jpg'.format(brojac))
      im2.save(img_path)       

      #Get text from image
      #im2.filter(ImageFilter.SHARPEN)
      imgForGS = cv2.imread(img_path, cv2.IMREAD_COLOR) 
      grayscale_img = convert_grayscale(imgForGS)
      cv2.imwrite(os.path.join(PATH_TO_RESULT_IMAGES_DIR, 'image{}-gs.jpg'.format(brojac)), grayscale_img)       
      imgForCR = cv2.imread(os.path.join(PATH_TO_RESULT_IMAGES_DIR, 'image{}-gs.jpg'.format(brojac)), cv2.IMREAD_COLOR) 
      text = Character_recognition(imgForCR)
      #pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
      #text = pytesseract.image_to_string(grayscale_img)
      print('image{}.jpg Text: '.format(brojac), text)

      brojac = brojac + 1  

