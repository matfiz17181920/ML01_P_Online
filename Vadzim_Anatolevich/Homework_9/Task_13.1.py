#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Импорт библиотек

import cv2;
import matplotlib.pyplot as plt;
import numpy as np;

#Блок функций

def extract_features_function(image_file, detector_type, mask = None):
    try:
        if detector_type != "sift" and detector_type != "orb":
            raise AttributeError("Детектора такого типа не существует! Доступные типы детекторов: 'sift', 'orb'");
        if detector_type == "sift":
            detector_type = cv2.SIFT_create();
        if detector_type == "orb":
            detector_type = cv2.ORB_create();
        key_points, descriptor = detector_type.detectAndCompute(image_file, mask);
    except:
        raise BaseException("Ошибка аттрибутов функции 'extract_features_function()'!");
    return key_points, descriptor;

def matches_features_function(descripor_1, descriptor_2, matching_type, detector_type, sort = True, k = 2):
    try:
        if matching_type == "BF":
            if detector_type != "sift" and detector_type != "orb":
                raise AttributeError("Детектора такого типа не существует! Доступные типы детекторов: 'sift', 'orb'");
            if detector_type == "sift":
                matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck = False);
            if detector_type == "orb":
                matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck = False);
            try:
                matches = matcher.knnMatch(descriptor_1, descriptor_2, k = k);
            except:
                raise BaseException("Измените тип детектора для изображений на 'orb'"); 
        elif matching_type != "BF":
            raise TypeError("Метода поиска такого типа не существует! Доступные типы методов поиска: 'BF'");      
        if sort:
            matches = sorted(matches, key = lambda x:x[0].distance);
    except:     
        raise BaseException("Ошибка аттрибутов функции 'matches_features_function()'!");
    return matches;

def filter_matches_function(matches, distance_threshold):
    try:
        filtered_matches = [];
        for a, b in matches:
            if a.distance <= distance_threshold * b.distance:
                filtered_matches.append(a);
    except:
        raise BaseException("Ошибка аттрибутов функции 'filter_matches_function()'!");        
    return filtered_matches;

def visualize_matches_function(image_1, key_points_1, image_2, key_points_2, filtered_matches):
    try:
        image_matches = cv2.drawMatches(image_1, key_points_1, image_2, key_points_2, filtered_matches, None, flags = 2);
        plt.figure(figsize = (16, 6), dpi = 100);
        plt.imshow(image_matches);
        plt.axis("off");
        plt.savefig("Combined_matches_image.jpeg", bbox_inches = "tight", pad_inches = 0);
        print("Комбинированное изображение сохранено в директорию проекта как растровое изображение: 'Combined_matches_image.jpeg'");
    except:
        raise BaseException("Ошибка аттрибутов функции 'visualize_matches_function()'!");

def transform_matrix_function(transform_values_1, transform_values_2, method):
    try:
        transform_matrix = cv2.estimateAffine2D(np.array(transform_values_2), np.array(transform_values_1),method = method)[0];    
        if transform_matrix is not None:
            dx = transform_matrix[0, 2];
            dy = transform_matrix[1, 2];
            da = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0]);
        else:
            dx = dy = da = 0;
    except:
        raise BaseException("Ошибка вычисления dx, dy, da и матрицы преобразования!");
    return [dx, dy, da], transform_matrix;

def decompose_image_affine_matrix_function(image_afine_transform_matrix):
    try:
        if image_afine_transform_matrix.shape != (3, 3) or image_afine_transform_matrix.dtype != np.float64:
            raise ValueError("Неверная размерность матрицы! Матрица должна иметь размерность 3х3");
        R = image_afine_transform_matrix[:2, :2];
        U, W, Vt = np.linalg.svd(R);
        rotation = np.dot(U, Vt);
        scaling = np.diag(W);
        translation = image_afine_transform_matrix[:2, 2:];
    except:
        raise BaseException("Ошибка декомпозиции матрицы афинного преобразования!");
    return rotation, translation, scaling;

def recompose_image_afine_transform_matrix_function(rotation, translation, scaling):
    try:
        rotation_scaling = np.dot(rotation, scaling);
        recompose_image_afine_transform_matrix = np.eye(3);
        recompose_image_afine_transform_matrix[:2, :2] = rotation_scaling;
        recompose_image_afine_transform_matrix[:2, 2] = translation.flatten();
    except:
        raise BaseException("Ошибка рекомпозиции матрицы афинного преобразования!");
    return recompose_image_afine_transform_matrix;

def get_angle_value_function(rotation):
    return np.arctan2(rotation[1, 0], rotation[0, 0]);

def get_scale_value_function(scaling):
    return np.sqrt(scaling[0, 0] ** 2 + scaling[1, 1] ** 2);

def get_translation_value_function(translation):
    return translation;

#Блок кода

image_1 = cv2.imread("Image_1.jpeg");
image_2 = cv2.imread("Image_2.jpeg");

key_points_1, descriptor_1 = extract_features_function(image_1, "sift");
key_points_2, descriptor_2 = extract_features_function(image_2, "sift");

matches_between_images = matches_features_function(descriptor_1, descriptor_2, "BF", "sift", sort = True);
print("Число совпадений ключевых точек до фильтрации:", len(matches_between_images));
filterd_matches_between_images = filter_matches_function(matches_between_images, 0.345);
print("Число совпадений ключевых точек после фильтрации:", len(filterd_matches_between_images));

visualize_matches_function(image_1, key_points_1, image_2, key_points_2, filterd_matches_between_images);

transform_values_1 = np.float32([key_points_1[m.queryIdx].pt for m in filterd_matches_between_images]).reshape(-1, 1, 2);
transform_values_2 = np.float32([key_points_2[m.trainIdx].pt for m in filterd_matches_between_images]).reshape(-1, 1, 2);

dx_dy_da , transform_matrix = transform_matrix_function(transform_values_1, transform_values_2, cv2.LMEDS);

print("Значения dx, dy, da: \n", dx_dy_da);
print("Матрица преобразования: \n", transform_matrix);

image_afine_transform_matrix =  np.append(transform_matrix,np.array([0,0,1])).reshape(3,3);
print("Афинная матрица преобразования: \n", image_afine_transform_matrix);

rotation, translation, scaling = decompose_image_affine_matrix_function(np.linalg.inv(image_afine_transform_matrix));
print("Значения матрицы поворота, вектора трансляции и матрицы масштабирования афинной матрицы преобразований: \n", rotation.shape, translation.shape, scaling.shape);
print("Исходная афинная матрица трансформации: \n", np.linalg.inv(image_afine_transform_matrix));
recompose_image_afine_transform_matrix = recompose_image_afine_transform_matrix_function(rotation, translation, scaling);
print("Восстановленная афинная матрица преобразований: \n", recompose_image_afine_transform_matrix);
image_afine_transform_matrix_equality_check = np.allclose(image_afine_transform_matrix, recompose_image_afine_transform_matrix);
print("Равенство исходной и восстановленной афинных матриц преобразований: \n");
if image_afine_transform_matrix_equality_check == True:
    print("Матрицы эквивалентны");
if image_afine_transform_matrix_equality_check == False:
    print("Матрицы не эквивалентны");

get_angle_value = get_angle_value_function(rotation);
get_translation_value = get_translation_value_function(translation);
get_scale_value = get_scale_value_function(scaling);
print("Явные значения поворота, вектора переноса и масштабирования афинной матрицы преобразований: \n", get_angle_value, get_translation_value, get_scale_value);    



