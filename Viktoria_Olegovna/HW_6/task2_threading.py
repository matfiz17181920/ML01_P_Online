import matplotlib.pyplot as plt
import concurrent.futures as cf
import time
import os

from tqdm import tqdm
from task2_helper import clear_dataset_folder, get_img_urls, create_plots_folder, download, IMGS_FOLDER, URL

"""
Реализовать с использованием потоков и процессов скачивание файлов из интернета. 
Список файлов для скачивания подготовить самостоятельно (например изображений, 
не менее 100 изображений или других объектов). Сравнить производительность с последовательным методом. 
Сравнивть производительность Thread и multiprocessing решений. Попробовать подобрать 
оптимальное число потоков/процессов. 
"""

# download images in treads
def multithreading_downloading():
    time_list = list()

    for threads_count in tqdm(range(50, MAX_THREADS_COUNT + 1, 50)):
        time_start = time.time()
        futures = list()

        clear_dataset_folder()
        with cf.ThreadPoolExecutor(max_workers=threads_count) as executor:
            for i in range(len(img_urls)):
                futures.append(executor.submit(download, url=img_urls[i], idx=i))

            for _ in cf.as_completed(futures):
                pass

        time_end = time.time()
        time_delta = time_end - time_start
        time_list.append(time_delta)

        print('Threads count =', threads_count, 'Download', len(os.listdir(IMGS_FOLDER)), 'images\nTakes', time_delta, 'sec\n')

    plt.plot(range(50, MAX_THREADS_COUNT + 1, 50), time_list)
    plt.title('Dependence time and threads count')
    plt.grid()
    plt.xlabel('threads count')
    plt.ylabel('time,sec')
    plt.savefig('plots/threading_plot.png')


if __name__ == '__main__':
    MAX_THREADS_COUNT = 200

    img_urls = get_img_urls(URL)
    print('Get', len(img_urls), 'images links')

    create_plots_folder()
    multithreading_downloading()


"""
output:

Get 213 images links
Threads count = 50 Download 213 images
Takes 1.789729356765747 sec

Threads count = 100 Download 213 images
Takes 1.0584654808044434 sec

Threads count = 150 Download 213 images
Takes 1.1836414337158203 sec

Threads count = 200 Download 213 images
Takes 1.157531499862671 sec

Оптимальным количеством потоков можно считать количество задач, 
которые возможно выполнить независимо друг от друга.
"""
