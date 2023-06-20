from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import math

from lib.external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from utils.debugger import Debugger

NON_MAX_SUP = True


class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  results_time = {'tot': 0, 'load': 0, 'pre': 0, 'net': 0, 'dec': 0, 'post': 0, 'merge': 0}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    results_all = []
    for x1, y1, image_cropped in crop_image_sliding_window(pre_processed_images['image'][0]):
      ret_cropped = detector.run(image_cropped)
      results_cropped = map_cropped_detections(ret_cropped['results'], x1, y1)
      results_all.extend(results_cropped[1])
      for key in results_time.keys():          
        results_time[key] = results_time[key] + ret_cropped[key]

    ret_original = detector.run(pre_processed_images)
    results_all.extend(ret_original['results'][1])
    for key in results_time.keys():          
      results_time[key] = results_time[key] + ret_original[key]

    if NON_MAX_SUP == True:
      ret = non_maximum_suppression(results_all)
      ret = {0: ret}
    else:
      ret = {0: results_all}
    
    results[ind] = ret
  
    # ret = detector.run(pre_processed_images)
    # results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(results_time[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}

  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    results_time = {'tot': 0, 'load': 0, 'pre': 0, 'net': 0, 'dec': 0, 'post': 0, 'merge': 0}
    results_all = []
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      for x1, y1, image_cropped in crop_image_sliding_window(image):
        ret_cropped = detector.run(image_cropped)
        results_cropped = map_cropped_detections(ret_cropped['results'], x1, y1)
        results_all.extend(results_cropped[1])
        for key in results_time.keys():          
          results_time[key] = results_time[key] + ret_cropped[key]

      ret_original = detector.run(image)
      results_all.extend(ret_original['results'][1])
      for key in results_time.keys():          
          results_time[key] = results_time[key] + ret_original[key]

      if NON_MAX_SUP == True:
          ret = non_maximum_suppression(results_all)
          ret = {0: ret}
      else:
        ret = {0: results_all}
    
    results[img_id] = ret
    # debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3),
    #                 theme=opt.debugger_theme)
    # show_results(debugger, image, ret, opt)

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(results_time[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  
  dataset.run_eval(results, opt.save_dir)

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
            yield (x, y, result_image)

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

def show_results(debugger, image, results, opt):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, opt.num_classes + 1):
        for bbox in results[j-1]:
          if bbox[4] > opt.vis_thresh:
            debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=True)


if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)