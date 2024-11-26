# Homework 06
<hr>


## Цель работы
1. Изучить основные принципы работы ***потоков*** и ***процессов*** в `Python`.
2. Изучить основные принципы работы с ***декораторами*** в `Python`.


## Задание
1. Реализовать с использованием потоков и процессов скачивание файлов из интернета. 
Список файлов для скачивания подготовить самостоятельно (например изображений, не менее 100 изображений или других объектов). 
Сравнить производительность с последовательным методом. 
Сравнить производительность Thread и multiprocessing решений. 
Попробовать подобрать оптимальное число потоков/процессов. 
2. Написать декоратор, оптимизирующий работу декорируемой функции. 
Декоратор должен сохранять результат работы функции на ближайшие три запуска и вместо выполнения функции возвращать сохранённый результат. 
После трёх запусков функция должна вызываться вновь, а результат работы функции — вновь кешироваться.

   
## Описание этапов разработки программы

### 1. Работа с потоками и процессами.

+ Рассмотрим простой пример, который циклически (10 циклов) осуществляет скачивание _200_ изображений по заданному `URL`
с последующим их сохранением. Для отправки запросов и загрузки данных будем использовать библиотеку `requests`.
Листинг программы приведен в файле `HW_06_requests.py`:
```
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
```
В результате выполнения получим:
```
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
```
Следует отметить, что данный код выполняется синхронно. 
Поэтому его выполнение будет зависеть от скорости интернет-соединения и отклика сервера.

+ Теперь рассмотрим пример, когда внутри одного процесса можно выполнять несколько потоков параллельно.
В Python для работы с потоками используется стандартный модуль `threading`, 
который предоставляет _API_ для создания и управления потоками.
Многопоточность позволяет программе выполнять несколько операций одновременно.
Листинг программы приведен в файле `HW_06_thread.py`:
```
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
```
Ниже приведен результат выполнения программного кода для разного числа одновременно работающих потоков (переменная `max_threads`):
```
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
```
Как видно, многопоточность значительно ускоряет процесс скачивания, поскольку файлы скачиваются параллельно. 
Многопоточность эффективна, когда программа ожидает завершения операций ввода-вывода (`I/O-bound` - сетевые запросы, чтение и запись файлов, работа с базами данных и т.д.).
В таких случаях, пока одни потоки "ожидают" завершения операций, другие потоки продолжают выполнение.
При увеличении числа потоков время скачивания файлов значительно уменьшается (до определенного момента).
Однако слишком большое количество потоков может перегрузить сервер или сеть.


+ Рассмотрим еще один вариант реализации, где будем комбинировать асинхронный подход с параллельным выполнением процессов. 
Асинхронность позволяет выполнять операции, не блокируя выполнение программы. 
_Miltiprocessing_ предоставляет возможность параллельного выполнения кода с использованием нескольких процессов.
Для асинхронных HTTP-запросов используется библиотека `aiohttp`, а для параллельного выполнения - `multiprocessing`.
Листинг программы приведен в файле `HW_06_multiproc.py`:
```
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
```
Ниже приведен результат выполнения программного кода для разного числа работающих процессов (переменная `num_processes`):
```
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
```
Как видно, многопроцессорность незначительно ускоряет процесс скачивания.
Многопроцессорность выгодна при обработке больших массивов данных, т.е. `CPU-bound` задач,
когда на ЦП приходится высокая нагрузка.


+ Теперь рассмотрим вариант реализации, где будем использовать асинхронный подход для параллельного выполнения процессов. 
Для асинхронных HTTP-запросов используется библиотека `aiohttp`. Для работы с параллельными задачами будем использовать модуль `asyncio`. 
Листинг программы приведен в файле `HW_06_aiohttp.py`:
```
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
```
Ниже приведен результат выполнения программного кода:
```
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
```
Как видно, асинхронные запросы позволяют эффективно скачивать файлы параллельно, что значительно ускоряет процесс.
<hr>


### 2. Работа с декораторами.
- Декораторы в Python — это функции, которые изменяют или расширяют поведение других функций или методов без изменения их исходного кода. 
- Напишем собственный декоратор, который будет сохранять результат работы функции на ближайшие три запуска и вместо выполнения функции возвращать сохранённый результат. 
После трёх запусков функция будет вызываться вновь, а результат работы функции — вновь кешироваться.
- При создании собственного декоратора удобно использовать готовую функцию `functools.wraps`, которая копирует метаданные исходной функции в обёртку.
Это позволяет сохранить инкапсуляцию и улучшить читаемость кода.
- Листинг программы приведен в файле `HW_06_cache_01.py`:
```
from functools import wraps


def cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        cache_key = (func, args, tuple(kwargs.items()))

        print(f"Function called {wrapper.cnt} times")

        if wrapper.cnt < 4:
            wrapper.cnt += 1

            if cache_key in wrapper.cache:
                print("Cached data...")
                return wrapper.cache[cache_key]
            else:
                wrapper.cache[cache_key] = func(*args, **kwargs)
                return wrapper.cache[cache_key]
        else:
            wrapper.cnt = 1
            wrapper.cache[cache_key] = func(*args, **kwargs)
            return wrapper.cache[cache_key]

    wrapper.cache = {}
    wrapper.cnt = 0

    return wrapper


@cache
def expensive_function(a, b):
    print(f"Calculating data...")
    return a + b


if __name__ == "__main__":

    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")  # clear cache
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(3, 3)}\n")  # other *args
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")  # clear cache
```

В результате выполнения получим:
```
Function called 0 times
Calculating data...
Result: 5

Function called 1 times
Cached data...
Result: 5

Function called 2 times
Cached data...
Result: 5

Function called 3 times
Cached data...
Result: 5

Function called 4 times
Calculating data...
Result: 5

Function called 1 times
Cached data...
Result: 5

Function called 2 times
Calculating data...
Result: 6

Function called 3 times
Cached data...
Result: 5

Function called 4 times
Calculating data...
Result: 5

Function called 1 times
Cached data...
Result: 5
```
<hr>

## Структура репозитория
Структура проекта `HW_06` имеет следующий вид:
+ в каталоге **`pics`** (будет создан в результате выполнения программы) содержатся скачанные изображения;
+ в каталоге **`src`** содержатся файлы **`.py`** с исходным кодом:
    * в файле **`HW_06_requests.py`** находится код программы для работы с синхронными запросами (`requests`);
    * в файле **`HW_06_threads.py`** находится код программы для работы с поточными запросами (`threads`);
    * в файле **`HW_06_multiproc.py`** находится код программы для работы с процессами (`multiprocessing`);
    * в файле **`HW_06_aiohttp.py`** находится код программы для работы с асинхронными процессами и запросами (`asyncio`, `aiohttp`);
    * в файле **`sittings.py`** находятся константы и параметры настройки для остальных программных файлов.
    * в файле **`HW_06_cache_01.py`** находится программный код, реализующий заданный в задании декоратор.
+ в файле **`requirements.txt`** содержится список необходимых для установки библиотек.


## Выполнение программ
Для запуска программ необходимо выполнить:
```
python.exe -m pip install -r .\requirements.txt
python.exe .\<script_name>
```
<hr>
