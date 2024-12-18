import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_matches(image1, kp1, image2, kp2, match):
    
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    plt.figure(figsize=(16, 6), dpi=200,)
    plt.imshow(image_matches)

    plt.axis('off')

    plt.savefig('plot_without_axes.png', bbox_inches='tight', pad_inches=0)
    
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

img0 = cv2.imread("IMG_1.JPG")
img1 = cv2.imread("IMG_2.JPG")
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

kp0, des0 = extract_features(img0, 'sift')
kp1, des1 = extract_features(img1, 'sift')
matches = match_features(des0, des1, matching='BF', detector='sift', sort=True)
print('Number of matches before filtering:', len(matches))
matches = filter_matches_distance(matches, 0.4)
print('Number of matches after filtering:', len(matches))
visualize_matches(img0, kp0, img1, kp1, matches)

def decompose_affine_matrix(affine):
    if affine.shape != (3, 3) or affine.dtype != np.float64:
        raise ValueError("Invalid input matrix. Must be a 3x3 double matrix.")

    R = affine[:2, :2]
    U, W, Vt = np.linalg.svd(R)

    rotation = np.dot(U, Vt)
    scaling = np.diag(W)
    translation = affine[:2, 2:]

    return rotation, translation, scaling

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

src_pts = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
m , M = estimate_partial_transform(src_pts,dst_pts, cv2.LMEDS)
afine_transform_matrix =  np.append(M,np.array([0,0,1])).reshape(3,3)
print(m)
print(M)
print(afine_transform_matrix)

def apply_affine_transformation(point, affine_matrix):
    
    point_homogeneous = np.array([point[0], point[1], 1])
    transformed_point = np.dot(affine_matrix, point_homogeneous)
    return transformed_point[:2]


if img0 is None or img1 is None:
    print("Ошибка: не удалось загрузить изображения. Проверьте пути.")
    exit()

point = (2000, 2400)

new_point = apply_affine_transformation(point, afine_transform_matrix)

cv2.circle(img1, point, 30, (0, 255, 0), -1)

cv2.circle(img0, (int(new_point[0]), int(new_point[1])), 20, (0, 0, 255), -1)
cv2.namedWindow("f", cv2.WINDOW_NORMAL)
cv2.resizeWindow("f", 600, 1200)
cv2.namedWindow("s", cv2.WINDOW_NORMAL)
cv2.resizeWindow("s", 800, 600)

cv2.imshow("f", img0)
cv2.imshow("s", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()