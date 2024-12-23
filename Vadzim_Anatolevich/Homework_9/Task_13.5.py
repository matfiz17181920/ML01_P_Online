#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Блок импорта библиотек

import matplotlib.pyplot as plt;
import cv2;
import kornia as K;
import kornia.feature as KF;
import numpy as np;
import torch;
from kornia_moons.feature import *;

#Блок функций

def local_descriptor_function(image, key_points, model):
  with torch.no_grad():
    model.eval();
    timg = K.color.rgb_to_grayscale(K.image_to_tensor(image, False).float()) / 255.;
    lafs = laf_from_opencv_SIFT_kpts(key_points);
    affine = KF.LAFAffNetShapeEstimator(True);
    orienter = KF.LAFOrienter(32, angle_detector = KF.OriNet(True));
    orienter.eval();
    affine.eval();
    lafs_new = orienter(affine(lafs, timg), timg);
    patches = KF.extract_patches_from_pyramid(timg, lafs_new, 32);
    B, N, CH, H, W = patches.size();
    descs = model(patches.view(B * N, CH, H, W)).view(B * N, -1);
  return descs.detach().cpu().numpy();

def points_matching_function(image_1, image_2, model):
  image_1 = cv2.cvtColor(cv2.imread(image_1), cv2.COLOR_BGR2RGB);
  image_2 = cv2.cvtColor(cv2.imread(image_2), cv2.COLOR_BGR2RGB);
  detector_type = cv2.SIFT_create(8000);
  key_points_1 = detector_type.detect(image_1, None);
  key_points_2 = detector_type.detect(image_2, None);
  descriptor_1 = local_descriptor_function(image_1, key_points_1, model);
  descriptor_2 = local_descriptor_function(image_2, key_points_2, model);
  dists, idxs = KF.match_smnn(torch.from_numpy(descriptor_1), torch.from_numpy(descriptor_2), 0.95);
  tentatives = cv2_matches_from_kornia(dists, idxs);
  source_points = np.float32([ key_points_1[m.queryIdx].pt for m in tentatives ]).reshape(-1,2);
  destination_points = np.float32([ key_points_2[m.trainIdx].pt for m in tentatives ]).reshape(-1,2);
  F, inliers_mask = cv2.estimateAffine2D(source_points, destination_points);
  draw_params = dict(matchColor = (255,255,0), singlePointColor = None, matchesMask = inliers_mask.ravel().tolist(), flags = 2);
  image_output = cv2.drawMatches(image_1, key_points_1, image_2, key_points_2, tentatives, None, **draw_params);
  plt.figure();
  fig, ax = plt.subplots(figsize = (15, 15));
  ax.imshow(image_output, interpolation = 'nearest');
  print (f'{inliers_mask.sum()} ключевых точек найдено');
  plt.show();
  
#Блок кода

image_1 = 'Image_1.jpeg';
image_2 = 'Image_2.jpeg';
model = KF.HardNet(True);
points_matching_function(image_1, image_2, model);