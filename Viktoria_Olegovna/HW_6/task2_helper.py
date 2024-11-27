import os
import time
import shutil
import requests

from bs4 import BeautifulSoup
from urllib.request import urlopen

URL = 'https://ru.123rf.com/%D0%A4%D0%BE%D1%82%D0%BE-%D1%81%D0%BE-%D1%81%D1%82%D0%BE%D0%BA%D0%B0/%D1%80%D0%BE%D0%B1%D0%BE%D1%82-%D1%81%D0%BE%D0%B1%D0%B0%D0%BA%D0%B0.html?imgtype=0'
IMGS_FOLDER = os.path.join(os.path.dirname(__file__), 'data')

# get images urls from site
def get_img_urls(url: str) -> list:
    html_data = requests.get(url).text
    content = BeautifulSoup(html_data, 'html.parser')

    return [img['src'] for img in content.find_all('img')]

# remove previous dataset folder and create new
def clear_dataset_folder():
    if os.path.exists(IMGS_FOLDER):
        shutil.rmtree(IMGS_FOLDER)     # remove folder with content

    os.mkdir(IMGS_FOLDER)

# downloand one image by url
def download(url: str, idx: int):
    try:
        resource = urlopen(url)
        out_img = open(os.path.join(IMGS_FOLDER, 'img_' + str(idx) + '.png'), 'wb')
        out_img.write(resource.read())
        out_img.close()
    except:
        pass
