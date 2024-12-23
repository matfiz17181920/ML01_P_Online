#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Блок импорта библиотек

import cv2;
import os;
import numpy as np;
from tqdm import tqdm;
import plotly.graph_objects as go;

#Блок функций

def filter_matches_distance(matches, dist_threshold):
    filtered_match = [];
    for m, n in matches:
        if m.distance <= dist_threshold*n.distance:
            filtered_match.append(m);
    return filtered_match;

def match_features(des1, des2, matching = 'BF', detector = 'sift', sort = True, k = 2):
    if matching == 'BF':
        if detector == 'sift':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck = False);
        elif detector == 'orb':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck = False);
        matches = matcher.knnMatch(des1, des2, k = k);
    elif matching == 'FLANN':
        FLANN_INDEX_KDTREE = 1;
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5);
        search_params = dict(checks = 50);
        matcher = cv2.FlannBasedMatcher(index_params, search_params);
        matches = matcher.knnMatch(des1, des2, k = k);
        if sort:
            matches = sorted(matches, key = lambda x:x[0].distance);
    return matches;

def convertrMatrixRotatedToEuler(Q: np.array) -> np.array:
    try:
        sinp = -Q[2, 0];
        cosp = (Q[0, 0]**2 + Q[1, 0]**2)**0.5;
        pitch = np.arctan2(sinp, cosp);
        sinr = Q[2, 1] / cosp;
        cosr = Q[2, 2] / cosp;
        roll = np.arctan2(sinr, cosr);
        siny = Q[1, 0] / cosp;
        cosy = Q[0, 0] / cosp;
        yaw = np.arctan2(siny, cosy);
        return np.array([np.rad2deg(roll),np.rad2deg(pitch),np.rad2deg(yaw)]);
    except Exception as e:
        print(e)
        
def create_trajectory(poses):
    trajectory = [np.array([0, 0, 0])];
    current_pose = np.eye(4);
    pose_cam = [np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])];
    for R, t in poses:
        T = np.eye(4);
        T[:3, :3] = R;
        T[:3, 3] = t.T;
        current_pose = np.dot(current_pose, T);
        trajectory.append(current_pose[:3, 3]);
        pose_cam.append(current_pose[:3, :3]);

    return np.array(trajectory),np.array(pose_cam);
 
#Блок кода

image_folder = "./frames";
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]);
image_folder = "./frames";
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]);
K = np.array([[3000 ,   0.    , 960], [  0.    , 3000 , 540 ], [  0.    ,   0.    ,   1.]], dtype = np.float32);
sift = cv2.SIFT_create();
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True);
poses = [];
positions = [np.array([0, 0, 0])];

for i in tqdm(range(len(image_files) - 1)):
    try:
       img1 = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE);
       img2 = cv2.imread(image_files[i + 1], cv2.IMREAD_GRAYSCALE);
       keypoints1, descriptors1 = sift.detectAndCompute(img1, None);
       keypoints2, descriptors2 = sift.detectAndCompute(img2, None);
       matches = match_features(descriptors1, descriptors2, matching = 'BF', detector = 'sift', sort = True);
       matches = filter_matches_distance(matches, 0.7);
       pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2);
       pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2);
       F, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.LMEDS, prob = 0.999, threshold = 1.0);  
       a, R, t, b = cv2.recoverPose(F, pts1, pts2, K, mask);
       poses.append((R, t));
       current_position = positions[-1];
       new_position = current_position + np.dot(R, t).T[0];
       positions.append(new_position);
    except:
       pass;
   
F, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.LMEDS, prob = 0.999, threshold = 1.0);      
a, R, t, b = cv2.recoverPose(F, pts1, pts2, K, mask);
trajectory,pose_cam = create_trajectory(poses);


fig = go.Figure();
fig.add_trace(go.Scatter3d(x = trajectory[:, 0], y = trajectory[:, 1], z = trajectory[:, 2], mode = 'lines+markers', marker = dict(size = 5, color = 'blue'), line = dict(color = 'blue', width = 2), name = 'Camera Trajectory'));

for i, (R, t) in enumerate(zip(pose_cam,trajectory)):
    camera_direction = R @ np.array([0, 0, 1]);
    camera_direction_end = t + camera_direction * 0.5;
    fig.add_trace(go.Scatter3d(x = [t[0], camera_direction_end[0]], y = [t[1], camera_direction_end[1]], z = [t[2], camera_direction_end[2]], mode = 'lines', line = dict(color = 'green', width = 2),showlegend = False if i > 0 else True));

fig.update_layout(title = 'Camera Motion Trajectory', scene = dict(xaxis_title = 'X-axis', yaxis_title = 'Y-axis', zaxis_title = 'Z-axis'), showlegend = True);
fig.show();
fig.write_html("camera_trajectory.html");

