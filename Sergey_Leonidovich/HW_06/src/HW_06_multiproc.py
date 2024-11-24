"""
ЗАДАНИЕ: Реализовать с использованием потоков и процессов скачивание файлов из интернета.
"""

"""
Реализация с использованием multiprocessing и библиотеки aiohttp.
"""

import os
import aiohttp
import asyncio
from multiprocessing import Process, Queue
from time import time
from settings import DIRS_PATH, SRC_URL, IMG_NUM, DOWNLOAD_NUM


async def get_img(session: aiohttp.ClientSession, url: str, filename: str):
    """Download image & save to file"""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(filename, "wb") as f:
                    f.write(await response.read())
                # print(f'Image {filename} downloaded')
            else:
                print(f"Failed to download image {filename}")
    except Exception as e:
        print(f"Error downloading image {filename}: {e}")


async def download_files(task_queue):
    """Create tasks to download images"""
    async with aiohttp.ClientSession() as session:
        while not task_queue.empty():
            try:
                filename = task_queue.get_nowait()
                await get_img(session, SRC_URL, filename)
            except asyncio.QueueEmpty:
                break


def main():

    for step in range(DOWNLOAD_NUM):

        # Number of process
        num_processes = 50

        # Path for images
        dir_path = DIRS_PATH.format(step)
        os.makedirs(dir_path, exist_ok=True)

        # Create task queue
        task_queue = Queue()
        for i in range(IMG_NUM):
            filename = os.path.join(dir_path, f"image_{i}.jpg")
            task_queue.put(filename)

        # Begin time
        t1 = time()

        # Start downloads (run processes)
        processes = []
        for _ in range(num_processes):
            # Asynchronous download
            p = Process(target=asyncio.run(download_files(task_queue)), args=(task_queue, ))
            processes.append(p)
            p.start()

        # Wait until all process stopped
        for p in processes:
            p.join()

        # End time
        print(f"Step {step + 1}. Working time: {time() - t1}")


if __name__ == "__main__":
    main()

    """
    # 2 process
    Step 1. Working time: 18.365092515945435
    Step 2. Working time: 21.656923294067383
    Step 3. Working time: 25.77588129043579
    Step 4. Working time: 18.577518463134766
    Step 5. Working time: 20.38513970375061
    Step 6. Working time: 17.465370893478394
    Step 7. Working time: 19.176352977752686
    Step 8. Working time: 23.476283073425293
    Step 9. Working time: 18.801722288131714
    Step 10. Working time: 20.177504301071167
    
    # 4 process
    Step 1. Working time: 18.120879411697388
    Step 2. Working time: 18.829487323760986
    Step 3. Working time: 18.92191219329834
    Step 4. Working time: 16.678121328353882
    Step 5. Working time: 19.35806155204773
    Step 6. Working time: 19.363309621810913
    Step 7. Working time: 18.2320556640625
    Step 8. Working time: 18.683732509613037
    Step 9. Working time: 18.759902477264404
    Step 10. Working time: 19.220251083374023
    
    # 5 process
    Step 1. Working time: 19.212634801864624
    Step 2. Working time: 19.753718376159668
    Step 3. Working time: 19.39814519882202
    Step 4. Working time: 18.31148099899292
    Step 5. Working time: 20.044885635375977
    Step 6. Working time: 17.991882801055908
    Step 7. Working time: 19.479783296585083
    Step 8. Working time: 18.589377403259277
    Step 9. Working time: 18.99623131752014
    Step 10. Working time: 19.341691255569458
    
    # 8 process
    Step 1. Working time: 23.094714403152466
    Step 2. Working time: 23.24116015434265
    Step 3. Working time: 32.257829904556274
    Step 4. Working time: 20.337135314941406
    Step 5. Working time: 18.531171798706055
    Step 6. Working time: 18.27836012840271
    Step 7. Working time: 17.812700033187866
    Step 8. Working time: 18.241811752319336
    Step 9. Working time: 49.3484423160553
    Step 10. Working time: 18.901888847351074
    """
