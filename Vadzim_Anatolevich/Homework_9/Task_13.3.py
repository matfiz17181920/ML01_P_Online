#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Блок импорта библиотек

import cv2;
import numpy as np;

# Блок функций

def track_video_function(input_video, output_video):
    print("Начало обработки видеопотока...");
    capture = cv2.VideoCapture(input_video);
    video_fps = capture.get(cv2.CAP_PROP_FPS);
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT));
    video_codec = cv2.VideoWriter_fourcc(*'mp4v');
    output = cv2.VideoWriter(output_video, video_codec, video_fps, (video_width, video_height));
    retur, first_frame = capture.read();
    first_frame = distortion_noise_function(first_frame);
    
    if not retur:
        print(f"Не удается прочитать видеофайл {input_video}");
        return;

    detector_type = cv2.ORB_create();
    keypoints, descriptors = detector_type.detectAndCompute(first_frame, None);

    if len(keypoints) > 0:
        initial_point = keypoints[240].pt;
    else:
        print("Ключевые точки не найдены");
        return;

    points_to_track = np.array([[initial_point]], dtype=np.float32);
    gray_map_initial = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY);
    center_point = (video_width // 2, video_height // 2);
    output.write(first_frame);
    trajectories = [];
    original_point = points_to_track[0][0];
    total_error_x = 0;
    total_error_y = 0;
    frame_count = 0;

    while True:
        retur, current_frame = capture.read();
        current_frame = distortion_noise_function(current_frame);
        
        if not retur:
            break;
        
        gray_map_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY);
        new_points, point_status, error = cv2.calcOpticalFlowPyrLK(gray_map_initial, gray_map_current, points_to_track, None);

        if point_status[0] == 1:
            tracked_point = new_points[0][0];
            trajectories.append((int(tracked_point[0]), int(tracked_point[1])));
            difference = np.array(tracked_point) - np.array(original_point);
            original_point = tracked_point;
            error_value_x = difference[0];
            error_value_y = difference[1];
            total_error_x += error_value_x;
            total_error_y += error_value_y;
            frame_count += 1;
            
            if len(trajectories) >= 3:
                source_points = np.array(trajectories[-3:], dtype=np.float32);
                destination_points = np.array([points_to_track[0][0], tracked_point, tracked_point], dtype=np.float32);
                M, inliers = cv2.estimateAffinePartial2D(source_points, destination_points);

                if M is not None:
                    points_to_track = cv2.transform(points_to_track, M);

            if len(trajectories) >= 3:
                for i in range(2, len(trajectories)):
                    cv2.line(current_frame, trajectories[i - 1], trajectories[i], (255, 0, 0), 2);

            if len(trajectories) > 3:
                cv2.circle(current_frame, (int(tracked_point[0]), int(tracked_point[1])), 10, (0, 0, 255), -1);

            cv2.circle(current_frame, (int(tracked_point[0]), int(tracked_point[1])), 10, (0, 0, 255), -1);
            cv2.drawMarker(current_frame, center_point, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2);
            end_point = (center_point[0] + 100, center_point[1] + 100);
            cv2.line(first_frame, center_point, end_point, (255, 0, 0), 2);
            draw_function(current_frame, tracked_point, trajectories, center_point, difference, total_error_x, total_error_y, frame_count);
            points_to_track = new_points.reshape(-1, 1, 2);
            
        else:
            break;

        output.write(current_frame);
        cv2.imshow("Video file point tracking in process...", current_frame);

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break;

        gray_map_initial = gray_map_current;

    capture.release();
    output.release();
    cv2.destroyAllWindows();
    print(f"Обработка видеопотока завершена. Обработанный файл {output_video} сохранен в директорию проекта");

def draw_function(current_frame, tracked_point, trajectories, center_point, difference, total_error_x, total_error_y, frame_count):
    coordinates_text = f"Координаты точки: X = {int(tracked_point[0])}, Y = {int(tracked_point[1])}";
    error_value = np.linalg.norm(np.array(trajectories[-1]) - np.array(tracked_point));
    error_text = f"Ошибка: {int(error_value)} пк";
    difference_text = f"Смещение опорной ключевой точки с учетом шумов: dX = {int(difference[0])}, dY = {int(difference[1])}";
    average_error_x_text = f"Средняя ошибка смещения с учетом шума: dX = {float(total_error_x / frame_count) if frame_count > 0 else 0} пк";
    average_error_y_text = f"Средняя ошибка смещения с учетом шума: dY = {float(total_error_y / frame_count) if frame_count > 0 else 0} пк";
    cv2.putText(current_frame, coordinates_text, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2);
    cv2.putText(current_frame, error_text, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2);
    cv2.putText(current_frame, difference_text, (20, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2);
    cv2.putText(current_frame, average_error_x_text, (20, 160), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2);
    cv2.putText(current_frame, average_error_y_text, (20, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2);

def distortion_noise_function(frame, distortion_level=0.65):
    noise = np.random.normal(0, 25, frame.shape).astype(np.uint8);
    distorted_frame = cv2.addWeighted(frame, 1 - distortion_level, noise, distortion_level, 0);
    return distorted_frame;

# Блок кода

track_video_function('Input_video.mp4', 'Output_video_(noise).mp4');