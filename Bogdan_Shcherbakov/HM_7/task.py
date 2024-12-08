import cv2

cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")

def rotation(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, matrix, (w, h))
    return rotated

def rotate_video(video_input, angle_per_second, video_output):
    if  video_input != "":
        cap = cv2.VideoCapture(video_input)
    else : 
        cap = cv2.VideoCapture(0)
    rotated_writer = cv2.VideoWriter(video_output, fourcc, fps, (width, height))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rotated_frame = rotation(frame, frame_idx * angle_per_second / fps)
        rotated_writer.write(rotated_frame)
        frame_idx += 1
        cv2.imshow("Original", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    rotated_writer.release()
    cv2.destroyAllWindows()

rotate_video("", 0, "input_video.avi")
rotate_video("input_video.avi", 10, "rotated_video.avi")
rotate_video("rotated_video.avi", -10, "output_video.avi")