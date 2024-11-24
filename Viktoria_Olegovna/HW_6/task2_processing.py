import multiprocessing
import time
import os

from task2_helper import get_img_urls, clear_dataset_folder, download, URL, IMGS_FOLDER

"""
Реализовать с использованием потоков и процессов скачивание файлов из интернета. 
Список файлов для скачивания подготовить самостоятельно (например изображений, 
не менее 100 изображений или других объектов). Сравнить производительность с последовательным методом. 
Сравнивть производительность Thread и multiprocessing решений. Попробовать подобрать 
оптимальное число потоков/процессов. 
"""

# download images in processes
def multiprocessing_download():
    for processes_count in range(1, MAX_PROCESSES_COUNT + 1):
        time_start = time.time()

        clear_dataset_folder()
        with multiprocessing.Pool(processes_count) as pool:
            pool.starmap(download, [[img_urls[i], i] for i in range(len(img_urls))])

        time_end = time.time()
        print('Processes count =', processes_count, 'Download', len(os.listdir(IMGS_FOLDER)), 'images\nTakes', time_end - time_start, 'sec\n')


if __name__ == '__main__':
    MAX_PROCESSES_COUNT = multiprocessing.cpu_count()

    img_urls = get_img_urls(URL)
    print('Get', len(img_urls), 'images links')
    multiprocessing_download()


"""
output:

Get 213 images links
Processes count = 1 Download 213 images
Takes 28.19198226928711 sec

Processes count = 2 Download 213 images
Takes 16.122265815734863 sec

Processes count = 3 Download 213 images
Takes 12.888463497161865 sec

Processes count = 4 Download 213 images
Takes 9.457687616348267 sec

Оптимальным количеством процессов можно считать количество ядер процесора, 
т к процессы выполняются параллельно, только когда запущены на разных ядрах.
"""
