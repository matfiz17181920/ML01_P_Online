#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2;

def rotate_function(frame, frame_rotating_angle):
    (height, width) = frame.shape[:2];
    video_center = (width // 2, height // 2);
    rotation_matrix = cv2.getRotationMatrix2D(video_center, frame_rotating_angle, 1.0);
    rotated_result = cv2.warpAffine(frame, rotation_matrix, (width, height));
    return rotated_result;

def reverse_video(result_video_path, reverse_video_path, rotating_angle):
    video_capture_r = cv2.VideoCapture(result_video_path);
    fps_count_r = video_capture_r.get(cv2.CAP_PROP_FPS);
    video_codec_r = cv2.VideoWriter_fourcc(*'XVID');
    reverse_video_creator = cv2.VideoWriter(reverse_video_path, video_codec_r, fps_count_r, (int(video_capture_r.get(3)), int(video_capture_r.get(4))));
    reverse_frame_angle = rotating_angle / fps_count_r;
    resulting_reversed_angle = 0;
    while video_capture_r.isOpened():
        ret, frame = video_capture_r.read();
        if ret != True:
            break;
        else:    
            resulting_reversed_angle = resulting_reversed_angle + reverse_frame_angle;
            rotated_reversed_frame = rotate_function(frame, resulting_reversed_angle);
            reverse_video_creator.write(rotated_reversed_frame);
    video_capture_r.release();
    reverse_video_creator.release();
    cv2.destroyAllWindows();

if __name__ == "__main__":
    rotating_angle = -10;
    result_video_path = 'Task_12_(video)_result.avi';
    reverse_video_path = 'Task_12_(video)_result_(reversed).avi';
    reverse_video(result_video_path, reverse_video_path, rotating_angle);
    