#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2;

def rotate_video(video_path, result_video_path):
    video_capture = cv2.VideoCapture(video_path);
    fps_count = video_capture.get(cv2.CAP_PROP_FPS);
    video_codec_type = cv2.VideoWriter_fourcc(*'XVID');
    result_video_creator = cv2.VideoWriter(result_video_path, video_codec_type, fps_count, (int(video_capture.get(3)), int(video_capture.get(4))));
    rotating_angle = 10;
    frame_rotating_angle = rotating_angle / fps_count;
    resulting_rotation_angle = 0;
    while video_capture.isOpened():
        ret, frame = video_capture.read();
        if ret != True:
            break;
        else:    
            resulting_rotation_angle = resulting_rotation_angle + frame_rotating_angle;
            rotated_frame = rotate_function(frame, resulting_rotation_angle);
            result_video_creator.write(rotated_frame);
    video_capture.release();
    result_video_creator.release();
    cv2.destroyAllWindows();

def rotate_function(frame, frame_rotating_angle):
    (height, width) = frame.shape[:2];
    video_center = (width // 2, height // 2);
    rotation_matrix = cv2.getRotationMatrix2D(video_center, frame_rotating_angle, 1.0);
    rotated_result = cv2.warpAffine(frame, rotation_matrix, (width, height));
    return rotated_result;

if __name__ == "__main__":
    video_path = 'Task_12_(video).mp4';
    result_video_path = 'Task_12_(video)_result.avi';
    rotate_video(video_path, result_video_path);

