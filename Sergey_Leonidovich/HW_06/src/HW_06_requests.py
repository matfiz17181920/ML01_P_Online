"""
ЗАДАНИЕ: Реализовать с использованием потоков и процессов скачивание файлов из интернета.
"""

"""
Реализация с использованием библиотеки requests.
"""

import os
import requests
from time import time
from settings import DIRS_PATH, SRC_URL, DOWNLOAD_NUM, IMG_NUM


def get_img(file_index, dir_path: str):
    """Download image & save to file"""
    filename = f"{dir_path}/image_{file_index}.jpg"
    try:
        response = requests.get(SRC_URL, stream=True)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            # print(f'Image {filename} downloaded')
        else:
            print(f"Failed to download image {filename}")
    except Exception as e:
        print(f"Error downloading image {filename}: {e}")


def main():
    for step in range(DOWNLOAD_NUM):

        # Path for images
        dirs_path = DIRS_PATH.format(step)
        os.makedirs(dirs_path, exist_ok=True)

        # Begin time
        t1 = time()

        # Download images
        for img in range(IMG_NUM):
            get_img(img, dirs_path)

        # End time
        print(f"Step {step + 1}. Working time: {time() - t1}")


if __name__ == "__main__":
    main()

    """
    Step 1. Working time: 27.404662132263184
    Step 2. Working time: 25.7896728515625
    Step 3. Working time: 25.375875234603882
    Step 4. Working time: 26.53287982940674
    Step 5. Working time: 26.765154361724854
    Step 6. Working time: 25.489325761795044
    Step 7. Working time: 27.716004133224487
    Step 8. Working time: 26.575775384902954
    Step 9. Working time: 27.512856245040894
    Step 10. Working time: 25.649073839187622
    """
