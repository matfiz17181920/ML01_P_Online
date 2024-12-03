import cv2
import numpy as np
import os

# Функция для конвертации видео в mp4
def convert_to_mp4(input_file, output_file):
    command = f'ffmpeg -i "{input_file}" "{output_file}"'
    os.system(command)

# Открытие исходного видеофайла
cap = cv2.VideoCapture(r'C:\Users\1neon\Desktop\CAT.mp4')

# Получение параметров видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Создание объекта для записи стабилизированного видео
stabilized_video_path = r'C:\Users\1neon\Desktop\stabilized_CAT.mp4'
out = cv2.VideoWriter(stabilized_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
center = (frame_width // 2, frame_height // 2)

# Обработка каждого кадра
angles = []
frames = []

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Угол поворота (15 градусов в секунду)
    angle = (15 * i / fps) % 360
    angles.append(angle)
    frames.append(frame)
    
    # Получение матрицы поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Применение поворота к кадру
    stabilized_frame = cv2.warpAffine(frame, rotation_matrix, (frame_width, frame_height), flags=cv2.INTER_LINEAR)

    # Запись стабилизированного кадра в видеофайл
    out.write(stabilized_frame)

# Завершение работы
cap.release()
out.release()

# Конвертация стабилизированного видео в mp4
convert_to_mp4(stabilized_video_path, stabilized_video_path)

# Создание объекта для записи исходного видео
restored_video_path = r'C:\Users\1neon\Desktop\restored_CAT.mp4'
out_original = cv2.VideoWriter(restored_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Запись исходных кадров в видеофайл
for frame in frames:
    out_original.write(frame)

# Завершение работы
out_original.release()

# Конвертация исходного видео в mp4
convert_to_mp4(restored_video_path, restored_video_path)

cv2.destroyAllWindows()
