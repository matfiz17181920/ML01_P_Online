import cv2
import os

output_video = 'output_town.mp4'
frame_rate = 30
frame_size = (640, 480)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, frame_size)

frames_folder = 'superpoint/video/town'

frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg'))])

for frame_file in frame_files:
    frame_path = os.path.join(frames_folder, frame_file)
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"Не удалось загрузить кадр: {frame_path}")
        continue
    
    frame = cv2.resize(frame, frame_size)
    
    video_writer.write(frame)

video_writer.release()
print(f"Видео успешно создано: {output_video}")
