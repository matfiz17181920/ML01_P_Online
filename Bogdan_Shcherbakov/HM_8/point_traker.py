import cv2
import numpy as np

def draw_points_on_frame(frame, point, color=(0, 0, 255), radius=5):
    cv2.circle(frame, point, radius, color, -1)
    return frame

def apply_affine_transformation(point, affine_matrix):
    point_homogeneous = np.array([point[0], point[1], 1])
    transformed_point = np.dot(affine_matrix, point_homogeneous)
    return transformed_point[:2]

def filter_matches_distance(matches, dist_threshold):
    filtered_match = []
    for m, n in matches:
        if m.distance <= dist_threshold*n.distance:
            filtered_match.append(m)
    return filtered_match

def match_features(des1, des2, matching='BF', detector='sift', sort=True, k=2):
    if matching == 'BF':
        if detector == 'sift':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=k)
    elif matching == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=k)
    if sort:
        matches = sorted(matches, key = lambda x:x[0].distance)
    return matches

def extract_features(image, detector, mask=None):
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()
    elif detector == 'surf':
        det = cv2.xfeatures2d.SURF_create()
    kp, des = det.detectAndCompute(image, mask)
    return kp, des

def estimate_partial_transform(cur_matched_kp, prev_matched_kp ,method):
    transform = cv2.estimateAffine2D(np.array(prev_matched_kp),
                                           np.array(cur_matched_kp),method=method)[0] #Тут как бы можно играться с тем как будут фильтроватся точки 
                                                                                          #cv2.LMEDS один из параметров который напрямую влияет    
    if transform is not None:
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])
    else:
        dx = dy = da = 0
    return [dx, dy, da], transform


def process_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_frame

def invert_affine_matrix(matrix):
    A = matrix[:2, :2]
    B = matrix[:2, 2]
    bottom_row = matrix[2, :]
    A_inv = np.linalg.inv(A) 
    B_inv = -np.dot(A_inv, B)
    inverse_matrix = np.eye(3)
    inverse_matrix[:2, :2] = A_inv
    inverse_matrix[:2, 2] = B_inv
    return inverse_matrix

def get_point_from_frame(frame):
    point = None
    def select_point(event, x, y, flags, param):
        nonlocal point
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)
    cv2.imshow("Выберите точку", frame)
    cv2.setMouseCallback("Выберите точку", select_point)

    while point is None:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return point

scale = 0.2
frame_count = 0

cap = cv2.VideoCapture("trimmed_town.mp4")
if not cap.isOpened():
    print("Ошибка: не удалось открыть видео.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'mp4v')

new_width = int(width * scale)
new_height = int(height * scale)

out = cv2.VideoWriter("trimmed_town(SIFT).mp4", codec, fps, (width, height))

ret, prev_frame = cap.read()
if not ret:
    print("Ошибка: не удалось прочитать первый кадр.")
    exit()
x, y = get_point_from_frame(prev_frame)
initial_point = (x * scale, y * scale)
prev_frame = cv2.resize(prev_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
prev_frame = process_frame(prev_frame)

current_point = initial_point

while True:
    ret, frame = cap.read()
    if not ret:
        break
    resize_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resize_frame = process_frame(resize_frame)
    resize_frame = cv2.GaussianBlur(resize_frame, (5, 5), sigmaX=1.5)
    
    kp0, des0 = extract_features(prev_frame, 'sift')
    kp1, des1 = extract_features(resize_frame, 'sift')
    matches = match_features(des0, des1, matching='BF', detector='sift', sort=True)
    matches = filter_matches_distance(matches, 0.6)
    
    src_pts = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    m , M = estimate_partial_transform(src_pts,dst_pts, cv2.LMEDS)
    afine_transform_matrix =  np.append(M,np.array([0,0,1])).reshape(3,3)
    
    current_point = apply_affine_transformation(current_point, invert_affine_matrix(afine_transform_matrix))
    
    # cv2.namedWindow("pointer", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("pointer", 1344, 756)

    cv2.circle(frame, (int(current_point[0]/scale), int(current_point[1]/scale)), 5, (0, 0, 255), -1)
    frame_count += 1
    print(frame_count)
    
    cv2.imshow("pointer", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    out.write(frame)
    prev_frame = resize_frame.copy()
    
    
cap.release()
out.release()
cv2.destroyAllWindows()