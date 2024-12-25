import cv2
import numpy as np
import plotly.graph_objects as go
import time

# Параметры камеры (замените на свои параметры)
K = np.array([[1000, 0, 1600],
              [0, 1000, 1200],
              [0, 0, 1]])  # Матрица внутренней калибровки

def show_progress(current_frame, total_frames):
    progress = (current_frame / total_frames) * 100
    print(f"Progress: {progress:.2f}%", end='\r')

# Функция для применения фильтра Собеля
def apply_sobel_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Производная по X
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Производная по Y
    sobel = cv2.magnitude(sobelx, sobely)  # Общая амплитуда градиента
    sobel = cv2.convertScaleAbs(sobel)  # Преобразование в uint8
    return sobel

# Траектории
def create_trajectory(poses):
    trajectory = [np.array([0, 0, 0])]
    current_pose = np.eye(4)
    pose_cam = [np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])]

    for R, t in poses:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.T
        current_pose = np.dot(current_pose, T)
        trajectory.append(current_pose[:3, 3])
        pose_cam.append(current_pose[:3, :3])

    return np.array(trajectory),np.array(pose_cam)

# Функция для обработки видео
def process_video(video_path, 
                  scale=1.0, 
                  use_sobel=True, 
                  frame_skip=1, 
                  flann_trees=5, 
                  flann_checks=50, 
                  ransac_threshold=1.0, 
                  lowe_ratio=0.75):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео")
        exit()

    # Выбор детектора SIFT
    sift = cv2.SIFT_create()
    flann_params = dict(algorithm=1, trees=flann_trees)
    search_params = dict(checks=flann_checks)
    flann = cv2.FlannBasedMatcher(flann_params, search_params)

    # Переменные для хранения траектории
    trajectory = [np.array([0, 0, 0])]
    orientations = []

    # Чтение первого кадра
    ret, prev_frame = cap.read()
    if not ret:
        print("Ошибка: не удалось считать первый кадр")
        cap.release()
        exit()

    if scale != 1.0:
        prev_frame = cv2.resize(prev_frame, None, fx=scale, fy=scale)

    if use_sobel:
        prev_frame = apply_sobel_filter(prev_frame)

    prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)

    # Инициализация положения и ориентации
    poses = []
    positions = [np.array([0, 0, 0])]

    # Начало обработки кадров
    start_time = time.time()
    while cap.isOpened():
        # Пропуск кадров
        for _ in range(frame_skip):
            ret, _ = cap.read()
            if not ret:
                break

        # Чтение следующего кадра
        ret, frame = cap.read()
        if not ret:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        if use_sobel:
            frame = apply_sobel_filter(frame)

        kp, des = sift.detectAndCompute(frame, None)

        # Сопоставление особенностей с помощью FLANN
        matches = flann.knnMatch(prev_des, des, k=2)

        # Применение правила Lowe для фильтрации совпадений
        good_matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]

        # Извлечение точек для соответствий
        if len(good_matches) > 0:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])

            # Вычисление матрицы Essential
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_threshold)

            # Восстановление положения и ориентации камеры
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask)

            # Сохраняем траекторию и ориентацию 
            poses.append((R, t))
            
            # Обновить позицию камеры
            current_position = positions[-1]
            new_position = current_position + np.dot(R, t).T[0]
            positions.append(new_position)

        # Обновление предыдущих кадров и точек
        current_frame += frame_skip
        show_progress(current_frame, total_frames)
        prev_frame = frame
        prev_kp, prev_des = kp, des

    # Завершаем обработку видео
    cap.release()

    # Преобразование траектории и направлений в массивы
    trajectory, orientations = create_trajectory(poses)

    # Построение 3D-графика
    fig = go.Figure()

    # Траектория камеры
    fig.add_trace(go.Scatter3d(
    x=trajectory[:, 0],
    y=trajectory[:, 1],
    z=trajectory[:, 2],
    mode='lines+markers',
    marker=dict(size=5, color='blue'),
    line=dict(color='blue', width=2),
    name='Camera Trajectory'
    ))

    # Направления взгляда камеры
    for i, (R, t) in enumerate(zip(orientations, trajectory)):
        # Направление камеры (ось Z камеры)
        camera_direction = R @ np.array([0, 0, 1])  # Направление оси Z камеры
        camera_direction_end = t + camera_direction * 0.5  # Конец вектора направления

        # Добавляем линию, представляющую направление камеры
        fig.add_trace(go.Scatter3d(
            x=[t[0], camera_direction_end[0]],
            y=[t[1], camera_direction_end[1]],
            z=[t[2], camera_direction_end[2]],
            mode='lines',
            line=dict(color='green', width=4),
            name=f'Camera Direction {i}' if i == 0 else None,
            showlegend=False if i > 0 else True
        ))

    # Настройка отображения
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="Camera Trajectory and View Directions",
        showlegend=True
    )

    # Сохранение графика в HTML-файл
    output_file = "camera_trajectory_and_view.html"
    fig.write_html(output_file)
    print(f"График сохранен в файл: {output_file}")

    end_time = time.time()
    print(f"Обработка видео заняла {end_time - start_time:.2f} секунд.")

# Вызов функции
process_video(
    video_path="test2.mp4",
    scale=0.5,  # Масштабирование видео (0.5 = уменьшение в 2 раза)
    use_sobel=True,  # Использовать фильтр Собеля
    frame_skip=2,  # Пропускать каждый второй кадр
    flann_trees=5,  # Параметр FLANN
    flann_checks=50,  # Параметр FLANN
    ransac_threshold=1.0,  # Порог для RANSAC
    lowe_ratio=0.75  # Коэффициент правила Lowe
)