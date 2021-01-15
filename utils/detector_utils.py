# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from utils import alertcheck
detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'my_model'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/labelmap.pbtxt'

NUM_CLASSES = 32
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    
    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess



def draw_box_on_image(num_products_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np,Orientation):


    color = None
    color0 = (255,0,0)
    color1 = (0,50,255)
    for i in range(num_products_detect):
        
        if (scores[i] > score_thresh):

            if classes[i] == 1:
                id = 'parachute hair cream'
            if classes[i] == 2:
                id = 'dabur anmol gold coconut oil'
            if classes[i] == 3:
                id = 'catch kashmiri mirch'
            if classes[i] == 4:
                id = 'patanjali chyawanprash'
            if classes[i] == 5:
                id = 'loreal smooth intense serum'
            if classes[i] == 6:
                id = 'emami mustard oil'
            if classes[i] == 7:
                id = 'dabur chyawanprash'
            if classes[i] == 8:
                id = 'good life hing'
            if classes[i] == 9:
                id = 'catch hing'
            if classes[i] == 10:
                id = 'ponds moisturizing lotion'
            if classes[i] == 11:
                id = 'catch garam masala'
            if classes[i] == 12:
                id = 'dabur chyawanprash 2x immunity'
            if classes[i] == 13:
                id = 'dabur honey'
            if classes[i] == 14:
                id = 'everest biryani masala'
            if classes[i] == 15:
                id = 'patanjali aloe vera gel'
            if classes[i] == 16:
                id = 'grofers kishmish'
            if classes[i] == 17:
                id = 'grofers room freshener'
            if classes[i] == 18:
                id = 'kiwi express shine sponge'
            if classes[i] == 19:
                id = 'garnier cleansing water'
            if classes[i] == 20:
                id = 'bc gulab jamun'
            if classes[i] == 21:
                id = 'bourn vita'
            if classes[i] == 22:
                id = 'catch chicken masala'
            if classes[i] == 23:
                id = 'everest pav bhaji masala'
            if classes[i] == 24:
                id = 'cinthol talcum powder'
            if classes[i] == 25:
                id = 'savemore glass cleaner'
            if classes[i] == 26:
                id = 'park avenue voyage perfume'
            if classes[i] == 27:
                id = 'orange coconut oil'
            if classes[i] == 28:
                id = 'everest hing'
            if classes[i] == 29:
                id = 'amul spray milk powder'
            if classes[i] == 30:
                id = 'yardley body spray'
            if classes[i] == 31:
                id = 'goodnight'
            if classes[i] == 32:
                id = 'fair and handsome'
            
            if i == 0: color = color0
            else: color = color1

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))


            cv2.rectangle(image_np, p1, p2, color , 3, 1)
            

            cv2.putText(image_np, ' '+str(i)+': '+id, (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
