import cv2

def trim_video(input_path, output_path, start_time, end_time):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Ошибка: не удалось открыть видео.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    if start_frame >= total_frames or end_frame > total_frames:
        raise ValueError("Указанные временные метки выходят за пределы длины видео.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1
        print(current_frame)

    cap.release()
    out.release()
    print(f"Обрезанное видео сохранено в: {output_path}")

trim_video("town.mp4", "trimmed_town.mp4", 265 , 315)