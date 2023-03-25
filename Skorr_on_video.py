import os
import time
import tensorflow as tf
import cv2
import numpy as np

from utils import label_map_util
from PIL import Image
from google.colab.patches import cv2_imshow
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from IPython.display import HTML
from base64 import b64encode

#Path to saved model  

PATH_TO_SAVED_MODEL = "/content/drive/MyDrive/Tensorflow2/Combined/inference_graph/saved_model"

# Load label map and obtain class names and ids
#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
category_index=label_map_util.create_category_index_from_labelmap("/content/drive/MyDrive/Tensorflow2/Combined/labelmap.pbtxt",use_display_name=True)

def getBB(list, bboxes):
    
    if 'Basketball' in list and 'Basketball Hoop' in list:
        BBallIndex = list.index("Basketball")
        BBallIndexNet = list.index("Basketball Hoop")

        BBallArray = bboxes[BBallIndex]
        BBallNetArray = bboxes[BBallIndexNet]

        ballYMin, ballXMin, ballYMax, ballXMax = BBallArray[0], BBallArray[1], BBallArray[2], BBallArray[3]
        netYMin, netXMin, netYMax, netXMax = BBallNetArray[0], BBallNetArray[1], BBallNetArray[2], BBallNetArray[3]

        bb1 = {'x1':ballXMin, 'x2':ballXMax, 'y1':ballYMin, 'y2':ballYMax}
        bb2 = {'x1':netXMin, 'x2':netXMax, 'y1':netYMin, 'y2':netYMax}
        return bb1, bb2
    
    elif 'Basketball' in list:
        print("Basketball Hoop not found in frame")
        bb1 = {'x1':0, 'x2':1, 'y1':0, 'y2':1}
        bb2 = {'x1':10, 'x2':11, 'y1':10, 'y2':11}
        return bb1, bb2 
    
    elif 'Basketball Hoop' in list:
        print("Basketball not found in frame")
        bb1 = {'x1':100, 'x2':101, 'y1':100, 'y2':101}
        bb2 = {'x1':20, 'x2':21, 'y1':20, 'y2':21}
        return bb1, bb2 

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def visualise_on_image(image, bboxes, labels, scores, thresh):
    (h, w, d) = image.shape
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(image, f"{label}: {int(score*100)} %", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    return image


if __name__ == '__main__':
    
    # Load the model
    print("Loading saved model ...")
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("Model Loaded!")
    points=0
    prev_iou = 0
    # Video Capture (video_file)
    video_capture = cv2.VideoCapture("/content/drive/MyDrive/Tensorflow2/Combined/vid.mp4")
    
    start_time = time.time()
    
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    #fps = int(video_capture.get(5))
    size = (frame_width, frame_height)
    
    #Initialize video writer
    result = cv2.VideoWriter('/content/drive/MyDrive/Tensorflow2/Combined/result.avi', cv2.VideoWriter_fourcc(*'MJPG'),15, size)

    while True:
      ret, frame = video_capture.read()
      if not ret:
          print('Unable to read video / Video ended')
          break
    
      frame = cv2.flip(frame, 1)
      image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      # The model expects a batch of images, so also add an axis with `tf.newaxis`.
      input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

      # Pass frame through detector
      detections = detect_fn(input_tensor)

      # Set detection parameters

      score_thresh = 0.4   # Minimum threshold for object detection
      max_detections = 3

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      scores = detections['detection_scores'][0, :max_detections].numpy()
      bboxes = detections['detection_boxes'][0, :max_detections].numpy()
      labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)
      print(labels)
      labels = [category_index[n]['name'] for n in labels]
      # print(scores)
      # print(bboxes)
      
      bb1, bb2 = getBB(labels, bboxes)
      iou = get_iou(bb1, bb2)
      
      print(iou)
      if iou == 1 and prev_iou==0:
        points+=1
      prev_iou = iou
      # Display detections
      visualise_on_image(frame, bboxes, labels, scores, score_thresh)

      # end_time = time.time()
      # fps = int(1/(end_time - start_time))
      # start_time = end_time
      # cv2.putText(frame, f"FPS: {fps}", (500,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (52, 235, 168), 2)
      #cv2_imshow(frame)
      cv2.putText(frame, f"Points: {points}", (500,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (52, 235, 168), 2)
      
      #Write output video
      result.write(frame)

    video_capture.release()
