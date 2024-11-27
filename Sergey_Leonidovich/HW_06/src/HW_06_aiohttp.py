"""
ЗАДАНИЕ: Реализовать с использованием потоков и процессов скачивание файлов из интернета.
"""

"""
Реализация с использованием асинхронных функций и библиотеки aiohttp.
"""

import asyncio
import aiohttp
import os
from time import time
from settings import SRC_URL, IMG_NUM, DOWNLOAD_NUM, DIRS_PATH


async def get_img(session: aiohttp.ClientSession, url: str, cur_num: int, dir_path: str):
    """Download image & save to file"""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                file_path = f"{dir_path}/image_{cur_num}.jpg"
                with open(file_path, "wb") as f:
                    f.write(await response.read())
                #print(f'Image {num} downloaded')
            else:
                print(f"Failed to download image {cur_num}")
    except Exception as e:
        print(f"Error downloading image {cur_num}: {e}")


async def download_images(dir_path: str):
    """Create tasks to download images"""
    async with aiohttp.ClientSession() as session:
        tasks = [get_img(session, SRC_URL, img_num, dir_path) for img_num in range(IMG_NUM)]
        await asyncio.gather(*tasks)


if __name__ == '__main__':

    for step in range(DOWNLOAD_NUM):

        # Path for images
        dirs_path = DIRS_PATH.format(step)
        os.makedirs(dirs_path, exist_ok=True)

        # Begin time
        t1 = time()

        # Start downloads
        asyncio.run(download_images(dirs_path))

        # End time
        print(f"Step {step + 1}. Working time: {time() - t1}")


    """
    Step 1. Working time: 1.208517074584961
    Step 2. Working time: 0.9981563091278076
    Step 3. Working time: 0.7199380397796631
    Step 4. Working time: 0.609403133392334
    Step 5. Working time: 0.6340553760528564
    Step 6. Working time: 0.6163640022277832
    Step 7. Working time: 0.77480149269104
    Step 8. Working time: 0.7060551643371582
    Step 9. Working time: 0.760800838470459
    Step 10. Working time: 0.6952114105224609
    """

