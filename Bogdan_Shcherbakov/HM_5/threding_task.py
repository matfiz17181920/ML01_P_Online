import os
import time
import requests
from threading import Thread
from multiprocessing import Pool

image_urls = [f"https://picsum.photos/200/300?random={i}" for i in range(1, 101)]

def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            filename = os.path.join("images", url.split("?random=")[-1] + ".jpg")
            with open(filename, "wb") as file:
                file.write(response.content)
    except Exception as e:
        print(f"Ошибка скачивания {url}: {e}")

def sequential_download(urls):
    for url in urls:
        download_image(url)

def threaded_download(urls, num_threads):
    threads = []
    for i in range(num_threads):
        thread_urls = urls[i::num_threads]
        thread = Thread(target=sequential_download, args=(thread_urls,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

def process_download(urls, num_processes):
    with Pool(num_processes) as pool:
        pool.map(download_image, urls)

def time_counter(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    return end - start

if __name__ == "__main__":
    
    for num_threads in [2, 4, 8, 16, 32]:
        threaded_time = time_counter(threaded_download, image_urls, num_threads)
        print(f"Скачивание используя {num_threads} потоков: {threaded_time:.2f} секунд")

    for num_processes in [2, 4, 8]:
        process_time = time_counter(process_download, image_urls, num_processes)
        print(f"Скачивание используя {num_processes} процессов: {process_time:.2f} секунд")
    
    sequential_time = time_counter(sequential_download, image_urls)
    print(f"Скачивание последовательно: {sequential_time:.2f} секунд")