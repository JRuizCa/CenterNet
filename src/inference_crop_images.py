from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import sys

import numpy as np
import math
import cv2

from opts import opts
from detectors.detector_factory import detector_factory
from utils.debugger import Debugger

CENTERNET_PATH = "/home/julia/TFM/CenterNet/src/lib/"
IMAGE_PATH = '/home/julia/TFM/CenterNet/data/barcode/test/13.jpg'
MODEL_PATH = "/home/julia/TFM/CenterNet/models/logs_2023-05-26-17-14/model_last.pth"
TASK = 'ctdet' 
NON_MAX_SUP = True
sys.path.insert(0, CENTERNET_PATH)

def crop_image_sliding_window(image, overlap_degree=0.4, window_size=512):
    height, width = image.shape[0:2]
    overlap = math.ceil(window_size * overlap_degree)
    step = window_size - overlap

    for y in range(0, height, step):
        for x in range(0, width, step):    
            y2 = y + window_size
            x2 = x + window_size
            zeros_image = np.zeros([window_size, window_size, 3], dtype=np.uint8)
            if y2 > height:
                y2 = height
            if x2 > width:
                x2 = width
            cropped_image = image[y:y2, x:x2]
            result_image = zeros_image.copy()
            result_image[:cropped_image.shape[0], :cropped_image.shape[1], :] = cropped_image
            yield (x, x2, y, y2, result_image)

def map_cropped_detections(detections, x1, y1):
    for j in range(1, len(detections)+1):
      if len(detections[j]) == 0:
        continue
      for detection in detections[j]:
        detection[0] = detection[0] + x1
        detection[1] = detection[1] + y1
        detection[2] = detection[2] + x1
        detection[3] = detection[3] + y1
    return detections

def show_results(debugger, image, results, opt):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, opt.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=True)

def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Bounding boxes are represented as [x_min, y_min, x_max, y_max].
    """
    x1, y1, x2, y2, score = bbox1
    x3, y3, x4, y4, score = bbox2
    
    intersection_width = max(0, min(x2, x4) - max(x1, x3))
    intersection_height = max(0, min(y2, y4) - max(y1, y3))
    intersection_area = intersection_width * intersection_height
    
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou

def non_maximum_suppression(bboxes, threshold=0, conf_threshold=0.3):
    bboxes_confidence = [x for x in bboxes if x[4] >= conf_threshold]
    scores = [x[4] for x in bboxes_confidence]
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    selected_indices = []
    idx = 0
    for i in sorted_indices:
        current_bbox = bboxes_confidence[i]
        selected_indices.append(i)
        for j in sorted_indices[idx + 1:]:
            iou = calculate_iou(current_bbox, bboxes_confidence[j])
            if iou > threshold:
                sorted_indices.remove(j)
        idx += 1
    selected_bboxes = [bboxes_confidence[i] for i in selected_indices]
    return selected_bboxes

def main():
    image = cv2.imread(IMAGE_PATH)
    results_original_image = []
    opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
    detector = detector_factory[opt.task](opt)
    print('Processing Crop Images')
    for x1, x2, y1, y2, image_cropped in crop_image_sliding_window(image):
        print(x1, x2, y1, y2)
        ret = detector.run(image_cropped)['results']
        results = map_cropped_detections(ret, x1, y1)
        results_original_image.extend(results[1])

    print('Processing Original Image')
    ret_original = detector.run(image)['results']
    results_original_image.extend(ret_original[1])

    if NON_MAX_SUP == True:
        res = non_maximum_suppression(results_original_image)
    else:
       res = results_original_image

    debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3),
                    theme=opt.debugger_theme)
    dict_results = {1: res}
    show_results(debugger, image, dict_results, opt)

if __name__ == '__main__':
    main()