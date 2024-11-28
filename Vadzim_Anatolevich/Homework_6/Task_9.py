# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:23:39 2024

@author: vdzmshacu
"""
import requests;
import os;
import time;
import threading;
from bs4 import BeautifulSoup;
from multiprocessing import Pool;
def image_urls(num_images=100):
    image_urls = set();
    while len(image_urls) < num_images:
        response = requests.get("https://en.wikipedia.org/wiki/Special:Random");
        if response.status_code != 200:
            continue;
        soup = BeautifulSoup(response.text, 'html.parser');
        for img in soup.find_all('img'):
            img_url = 'https:' + img['src'];
            if img_url not in image_urls:
                image_urls.add(img_url);
                if len(image_urls) >= num_images:
                    break;
    return list(image_urls);
url_list = image_urls(100)
def download(url):
    try:
        response = requests.get(url);
        if response.status_code == 200:
            file_name = os.path.join('images', url.split('/')[-1]);
            with open(file_name, "wb") as f:
                f.write(response.content);
            print(f"Скачан файл: {file_name}");
        else:
            print(f"Ошибка загрузки файла: {url}: Статус загрузки файла: {response.status_code}");
    except Exception as e:
        print(f"Ошибка скачивания файла: {url}: {e}");
os.makedirs('images', exist_ok=True);
def sequential(urls):
    for url in urls:
        download(url);
start_time = time.time();
sequential(url_list);
end_time = time.time();
print(f"Время последовательной загрузки файлов: {end_time - start_time} сек.");
def thread(urls):
    threads = [];
    for url in urls:
        thread = threading.Thread(target=download, args=(url,));
        threads.append(thread);
        thread.start();
    for thread in threads:
        thread.join();
start_time = time.time();
thread(url_list);
end_time = time.time();
print(f"Время потоковой загрузки файлов: {end_time - start_time} сек.");
def multiprocessing(urls):
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(download, urls);
start_time = time.time();
multiprocessing(url_list);
end_time = time.time();
print(f"Время загрузки файлов в мультипроцессинговом режиме: {end_time - start_time} сек.");