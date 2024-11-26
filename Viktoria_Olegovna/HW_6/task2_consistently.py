import os
import time

from task2_helper import get_img_urls, clear_dataset_folder, download, URL, IMGS_FOLDER

URL = 'https://ru.123rf.com/%D0%A4%D0%BE%D1%82%D0%BE-%D1%81%D0%BE-%D1%81%D1%82%D0%BE%D0%BA%D0%B0/%D1%80%D0%BE%D0%B1%D0%BE%D1%82-%D1%81%D0%BE%D0%B1%D0%B0%D0%BA%D0%B0.html?imgtype=0'
IMGS_FOLDER = os.path.join(os.path.dirname(__file__), 'data')

# download images consistently in circle
def consistently_downloading():
    time_start = time.time()
    clear_dataset_folder()

    for i in range(len(img_urls)):
        download(url=img_urls[i], idx=i)

    time_end = time.time()
    print('Download', len(os.listdir(IMGS_FOLDER)), 'images\nTakes', time_end - time_start, 'sec\n')


if __name__ == '__main__':
    img_urls = get_img_urls(URL)
    consistently_downloading()


"""
output:

Download 312 images
Takes 32.51130771636963 sec

"""
