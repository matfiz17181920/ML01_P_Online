"""
ЗАДАНИЕ: Реализовать с использованием потоков и процессов скачивание файлов из интернета.
"""

"""
Реализация с использованием потоков (Threads) и библиотеки requests.
"""

import os
import requests
from time import time
from concurrent.futures import ThreadPoolExecutor
from settings import DIRS_PATH, SRC_URL, DOWNLOAD_NUM, IMG_NUM
from HW_06_requests import get_img


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

        # Number of MAX threads
        max_threads = 5

        # Begin time
        t1 = time()

        # Use ThreadPoolExecutor for threads control
        with ThreadPoolExecutor(max_threads) as executor:
            executor.map(get_img, range(IMG_NUM), [dirs_path] * IMG_NUM)

        # End time
        print(f"Step {step + 1}. Working time: {time() - t1}")


if __name__ == "__main__":
    main()

    """
    # 5 threads
    Step 1. Working time: 5.772582054138184
    Step 2. Working time: 5.4377923011779785
    Step 3. Working time: 4.891822099685669
    Step 4. Working time: 4.822983026504517
    Step 5. Working time: 5.361803293228149
    Step 6. Working time: 4.921698093414307
    Step 7. Working time: 4.975452423095703
    Step 8. Working time: 4.939796686172485
    Step 9. Working time: 4.889312982559204
    Step 10. Working time: 5.047741413116455
        
    # 10 threads
    Step 1. Working time: 2.7775161266326904
    Step 2. Working time: 2.7543060779571533
    Step 3. Working time: 2.667933225631714
    Step 4. Working time: 2.710009813308716
    Step 5. Working time: 2.5603387355804443
    Step 6. Working time: 2.7641263008117676
    Step 7. Working time: 2.5673580169677734
    Step 8. Working time: 2.566572904586792
    Step 9. Working time: 2.652371644973755
    Step 10. Working time: 2.7001137733459473

    # 20 threads
    Step 1. Working time: 1.7689557075500488
    Step 2. Working time: 1.6545863151550293
    Step 3. Working time: 1.3344829082489014
    Step 4. Working time: 1.3223934173583984
    Step 5. Working time: 1.4570014476776123
    Step 6. Working time: 1.2976667881011963
    Step 7. Working time: 2.169520616531372
    Step 8. Working time: 1.3365814685821533
    Step 9. Working time: 1.3292458057403564
    Step 10. Working time: 1.32893705368042
    
    #50 threads
    Step 1. Working time: 1.3218984603881836
    Step 2. Working time: 0.7831265926361084
    Step 3. Working time: 0.8045196533203125
    Step 4. Working time: 0.8189098834991455
    Step 5. Working time: 0.7848331928253174
    Step 6. Working time: 0.808779239654541
    Step 7. Working time: 0.9959406852722168
    Step 8. Working time: 0.9783976078033447
    Step 9. Working time: 0.840277910232544
    Step 10. Working time: 0.8005197048187256
    
    #100 threads
    Step 1. Working time: 0.9819786548614502
    Step 2. Working time: 0.8713278770446777
    Step 3. Working time: 0.714165210723877
    Step 4. Working time: 0.760735034942627
    Step 5. Working time: 0.7780356407165527
    Step 6. Working time: 0.8546874523162842
    Step 7. Working time: 0.7801840305328369
    Step 8. Working time: 0.811227560043335
    Step 9. Working time: 0.8597137928009033
    Step 10. Working time: 0.8979663848876953
    
    # 200 threads
    Step 1. Working time: 0.9206016063690186
    Step 2. Working time: 0.7249929904937744
    Step 3. Working time: 0.8030824661254883
    Step 4. Working time: 0.8324763774871826
    Step 5. Working time: 0.8208682537078857
    Step 6. Working time: 0.8408036231994629
    Step 7. Working time: 0.7964110374450684
    Step 8. Working time: 0.8711690902709961
    Step 9. Working time: 0.7654509544372559
    Step 10. Working time: 0.7448806762695312
    """