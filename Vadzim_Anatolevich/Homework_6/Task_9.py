# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:23:39 2024

@author: vdmsa
"""
import requests
import os
import time
import threading
from bs4 import BeautifulSoup
from multiprocessing import Pool

# Шаг 1: Получаем URL изображений из случайных статей Википедии
def get_image_urls(num_images=100):
    image_urls = set()
    while len(image_urls) < num_images:
        response = requests.get("https://en.wikipedia.org/wiki/Special:Random")
        if response.status_code != 200:
            continue
        soup = BeautifulSoup(response.text, 'html.parser')

        # Поиск всех тегов <img>
        for img in soup.find_all('img'):
            img_url = 'https:' + img['src']  # Протокол обязательно добавляем
            if img_url not in image_urls:
                image_urls.add(img_url)
                if len(image_urls) >= num_images:
                    break

    return list(image_urls)

url_list = get_image_urls(100)

# Шаг 2: Функция для скачивания изображений
def download_file(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_name = os.path.join('images', url.split('/')[-1])  # Используем последнюю часть URL как имя файла
            with open(file_name, "wb") as f:
                f.write(response.content)
            print(f"Скачано: {file_name}")
        else:
            print(f"Ошибка загрузки {url}: Статус {response.status_code}")
    except Exception as e:
        print(f"Ошибка скачивания {url}: {e}")

# Создание директории для сохранения изображений
os.makedirs('images', exist_ok=True)

# 2.1 Последовательное скачивание
def sequential_download(urls):
    for url in urls:
        download_file(url)

start_time = time.time()
sequential_download(url_list)
end_time = time.time()
print(f"Sequential download time: {end_time - start_time} seconds")

# 2.2 Многопоточное скачивание
def threaded_download(urls):
    threads = []
    for url in urls:
        thread = threading.Thread(target=download_file, args=(url,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

start_time = time.time()
threaded_download(url_list)
end_time = time.time()
print(f"Threaded download time: {end_time - start_time} seconds")

# 2.3 Многопроцессорное скачивание
def multiprocessing_download(urls):
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(download_file, urls)

start_time = time.time()
multiprocessing_download(url_list)
end_time = time.time()
print(f"Multiprocessing download time: {end_time - start_time} seconds")