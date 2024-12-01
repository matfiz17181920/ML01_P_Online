import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Функция для скачивания одного файла
def download_file(url, folder=None):
    if folder is None:
        # По умолчанию сохраняем файлы на рабочий стол в папку "images"
        folder = os.path.join(os.path.expanduser("~"), "Desktop", "images")
    if not os.path.exists(folder):
        os.makedirs(folder)  # Создаем папку, если она не существует
    # Определяем имя файла и путь для сохранения
    local_filename = os.path.join(folder, url.split('/')[-1].replace('?', '_'))
    # Скачиваем файл с использованием библиотеки requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Проверяем, что запрос прошел успешно
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)  # Сохраняем файл по частям
    print(f"Скачано: {local_filename}")
    return local_filename

# Функция для скачивания файлов с использованием потоков
def download_files_in_threads(urls, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_file, urls)

# Функция для скачивания файлов с использованием процессов
def download_files_in_processes(urls, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_file, urls)

# Функция для генерации списка URL для тестирования
def generate_file_urls():
    return [f"https://via.placeholder.com/150?text={i}" for i in range(100)]

if __name__ == "__main__":
    urls = generate_file_urls()
    
    # Последовательное скачивание
    start_time = time.time()
    for url in urls:
        download_file(url)
    sequential_time = time.time() - start_time
    print(f"Sequential download time: {sequential_time:.2f} seconds")
    
    # Скачивание с использованием потоков
    max_workers_list = [2, 5, 10, 20]  # Разные числа потоков для тестирования
    for max_workers in max_workers_list:
        start_time = time.time()
        download_files_in_threads(urls, max_workers)
        thread_time = time.time() - start_time
        print(f"Threaded download time with {max_workers} workers: {thread_time:.2f} seconds")
    
    # Скачивание с использованием процессов
    for max_workers in max_workers_list:
        start_time = time.time()
        download_files_in_processes(urls, max_workers)
        process_time = time.time() - start_time
        print(f"Process download time with {max_workers} workers: {process_time:.2f} seconds")
