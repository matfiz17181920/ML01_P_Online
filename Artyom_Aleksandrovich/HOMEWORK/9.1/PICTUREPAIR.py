import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображений
img1 = cv2.imread('cat1.jpg')
img2 = cv2.imread('cat2.jpg')

# Преобразование изображений в градации серого
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

def extract_features(image, detector='sift'):
    """
    Находит ключевые точки и дескрипторы для изображения

    Аргументы:
    image -- изображение в градациях серого
    detector -- 'sift' или 'orb'. По умолчанию 'sift'

    Возвращает:
    kp -- список извлеченных ключевых точек (особенностей) на изображении
    des -- список дескрипторов ключевых точек на изображении
    """
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()
        
    kp, des = det.detectAndCompute(image, None)
    
    return kp, des

# Извлечение ключевых точек и дескрипторов с помощью SIFT
kp1_sift, des1_sift = extract_features(gray1, 'sift')
kp2_sift, des2_sift = extract_features(gray2, 'sift')

# Извлечение ключевых точек и дескрипторов с помощью ORB
kp1_orb, des1_orb = extract_features(gray1, 'orb')
kp2_orb, des2_orb = extract_features(gray2, 'orb')

# Функция для отображения ключевых точек
def draw_keypoints(image, keypoints, title):
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.show()

# Отображение ключевых точек для каждого метода
draw_keypoints(img1, kp1_sift, 'SIFT Keypoints on Image 1')
draw_keypoints(img2, kp2_sift, 'SIFT Keypoints on Image 2')

draw_keypoints(img1, kp1_orb, 'ORB Keypoints on Image 1')
draw_keypoints(img2, kp2_orb, 'ORB Keypoints on Image 2')

# Сравнение количества ключевых точек, найденных каждым методом
print(f"SIFT: {len(kp1_sift)} keypoints on Image 1, {len(kp2_sift)} keypoints on Image 2")
print(f"ORB: {len(kp1_orb)} keypoints on Image 1, {len(kp2_orb)} keypoints on Image 2")

# Реальные координаты точек на изображениях
pts1 = np.float32([[528, 579], [668, 551], [676, 732], [865, 912], [696, 918], [299, 1194]])
pts2 = np.float32([[76, 534], [308, 1159], [234, 690], [529, 579], [668, 551], [678, 735]])

# Вычисление матрицы афинного преобразования с использованием метода наименьших квадратов
M, _ = cv2.estimateAffine2D(pts1, pts2)
print("Матрица афинного преобразования:\n", M)

# Разложение матрицы на компоненты
U, S, Vt = np.linalg.svd(M[:2, :2])

# Угол поворота
theta = np.arctan2(U[1, 0], U[0, 0])
print("Угол поворота (в радианах):", theta)

# Вектор переноса
translation = M[:, 2]
print("Вектор переноса:", translation)

# Скалирование
scale = S
print("Скалирование:", scale)

# Вычисление обратной матрицы афинного преобразования
M_inv = cv2.invertAffineTransform(M)
print("Обратная матрица афинного преобразования:\n", M_inv)

# Применение обратного преобразования к исходным точкам для проверки
pts1_transformed = cv2.transform(np.array([pts2]), M_inv)
print("Преобразованные точки:\n", pts1_transformed)

# Функция для отображения изображений и соединения точек
def draw_matches(img1, img2, pts1, pts2):
    # Создаем изображение для отображения
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img_combined[:h1, :w1, :] = img1
    img_combined[:h2, w1:w1 + w2, :] = img2

    # Рисуем линии, соединяющие соответствующие точки
    for pt1, pt2 in zip(pts1, pts2):
        pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))
        cv2.line(img_combined, (int(pt1[0]), int(pt1[1])), pt2_shifted, (0, 255, 0), 2)
        cv2.circle(img_combined, (int(pt1[0]), int(pt1[1])), 5, (0, 0, 255), -1)
        cv2.circle(img_combined, pt2_shifted, 5, (0, 0, 255), -1)

    return img_combined

# Отображение изображений с соединенными точками
img_matches = draw_matches(img1, img2, pts1, pts2)
plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.show()
