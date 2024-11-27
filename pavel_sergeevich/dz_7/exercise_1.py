"""
Задание:
    Реализовать с использованием потоков и процессов скачивание файлов из интернета.
    Список файлов для скачивания подготовить самостоятельно (например изображений, не менее 100 изображений
    или других объектов). Сравнить производительность с последовательным методом. Сравнивть производительность
    Thread и multiprocessing решений. Попробовать подобрать оптимальное число потоков/процессов.

Выводы:
    Самым сложным было найти файлы для скачивания, по итогу было нйадено решение - скачать страничку, распарсить ее,
    достать теги img, в которых есть url к картинке. Выкачиваем картинку. Картинки скачивались тремя способами:
    последовательно, с помощью потоков и с помощью процессов. На последовательное скачивание требуется порядка 30-40
    секунд. На скачивание с использованием потоков - 6-7 секунд. Эксперименты показали, что оптимальным числом потоков
    будет число логических просессоров (в моем случае 8), причем если уменьшить его до 6, производительность падает
    не сильно, время скачивания примерно возрастает на 1 секунду. А вот если продолжить уменьшать количество потоков -
    производительность начинает резко падать. Увеличение количества потоков более 8 - прироста производительности
    не дает.

    Решение задачи через процессы показало себя несколько хуже. Оптимальное число процессов и зависимсоть
    производительности от числа процессов совпадает с результатами применения пула потоков. Однако результирующее время
    больше, т.к. само создание процесса весьма дорогостоящая операция.

"""
import random
import sys
import time
import shutil
import os

import concurrent.futures

import urllib.request
from bs4 import BeautifulSoup
import requests


def get_tags(url):
    # скачивание страничку и запихиваем ее на парсер
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # поиск тегов img (они содержат url картинки)
    image_tags = soup.find_all('img')

    print("Найденное количество картинок: " + str(len(image_tags)))

    return image_tags


def download(tag, path):
    # извлечение URL изображения из атрибута 'src' и выделяем его имя
    img_url = tag['src']

    filename = path + "/" + img_url.split('/')[-1]
    try:
        urllib.request.urlretrieve(img_url, filename)
    except Exception as e:
        print("Невозможно загрузить " + str(img_url) + ": " + str(e))

        return False
    return True


if __name__ == "__main__":
    base_url = "https://blog.studiominiboss.com/pixelart"
    download_dir = "download"

    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)

    tags_img = get_tags(base_url)

    # засекаем время для последовательного скачивания
    start = time.perf_counter()

    cnt = 0
    for tg in tags_img:
        if download(tg, download_dir):
            cnt += 1

    print("Скачено картинок: " + str(cnt))
    print("Время последовательного скачивания (секунды): " + str(time.perf_counter() - start))

    shutil.rmtree(download_dir)
    os.mkdir(download_dir)

    # ставлю задержку, чтобы пугать сервер, на всякий случай
    time.sleep(random.randint(3, 10))

    print("-" * 120)

    # засекаем время для скачивания с помощью потоков
    start = time.perf_counter()

    cnt = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for tg in tags_img:
            futures.append(executor.submit(download, tag=tg, path=download_dir))

        for future in concurrent.futures.as_completed(futures):
            if future.result():
                cnt += 1

    print("Скачено картинок: " + str(cnt))
    print("Время скачивания с помощью потоков (секунды): " + str(time.perf_counter() - start))

    # ставлю задержку, чтобы пугать сервер, на всякий случай
    time.sleep(random.randint(3, 10))

    print("-" * 120)

    # засекаем время для скачивания с помощью ProcessPoolExecutor
    start = time.perf_counter()

    # фактически костыль - BeautifulSoup строит большое дерево (html-страница) в котором очень много рекурсивных связей
    # в свою очередь при создании процесса в него передается pickle объект и проблема заключается в том, что
    # BeautifulSoup деревья не могут быть сереализованы
    sys.setrecursionlimit(25000)

    cnt = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for tg in tags_img:
            futures.append(executor.submit(download, tag=tg, path=download_dir))

        for future in concurrent.futures.as_completed(futures):
            if future.result():
                cnt += 1

    print("Скачено картинок: " + str(cnt))
    print("Время скачивания с помощью процессов (секунды): " + str(time.perf_counter() - start))

    print("-" * 120)

"""
Вывод программы

Найденное количество картинок: 100

Невозможно загрузить https://cdn-images-1.medium.com/max/600/1*8hOSIYTOWBg6ZZ5U9AkNcg.png: HTTP Error 403: Forbidden
Невозможно загрузить http://i.imgur.com/WQ6WdOF.gif: HTTP Error 429: Unknown Error
Невозможно загрузить https://i.imgur.com/cxz6GRE.gif: HTTP Error 429: Unknown Error
Невозможно загрузить https://px.srvcs.tumblr.com/impixu?T=1732135720&J=eyJ0eXBlIjoidXJsIiwidXJsIjoiaHR0cDo... (обрезано)
Скачено картинок: 96
Время последовательного скачивания (секунды): 31.5642403
------------------------------------------------------------------------------------------------------------------------
Невозможно загрузить https://cdn-images-1.medium.com/max/600/1*8hOSIYTOWBg6ZZ5U9AkNcg.png: HTTP Error 403: Forbidden
Невозможно загрузить http://i.imgur.com/WQ6WdOF.gif: HTTP Error 429: Unknown Error
Невозможно загрузить https://i.imgur.com/cxz6GRE.gif: HTTP Error 429: Unknown Error
Невозможно загрузить https://px.srvcs.tumblr.com/impixu?T=1732135720&J=eyJ0eXBlIjoidXJsIiwidXJsIjoiaHR0cDo... (обрезано)
Скачено картинок: 96
Время скачивания с помощью потоков (секунды): 6.536405299999998
------------------------------------------------------------------------------------------------------------------------
Невозможно загрузить https://cdn-images-1.medium.com/max/600/1*8hOSIYTOWBg6ZZ5U9AkNcg.png: HTTP Error 403: Forbidden
Невозможно загрузить http://i.imgur.com/WQ6WdOF.gif: HTTP Error 429: Unknown Error
Невозможно загрузить https://i.imgur.com/cxz6GRE.gif: HTTP Error 429: Unknown Error
Невозможно загрузить https://px.srvcs.tumblr.com/impixu?T=1732135720&J=eyJ0eXBlIjoidXJsIiwidXJsIjoiaHR0cDo... (обрезано)
Скачено картинок: 96
Время скачивания с помощью процессов (секунды): 7.665846199999997
------------------------------------------------------------------------------------------------------------------------
"""