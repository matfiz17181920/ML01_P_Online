"""
Задача:
Реализовать с использованием потоков и процессов скачивание файлов из интернета. 
Список файлов для скачивания подготовить самостоятельно (например изображений, не менее 100 изображений или других объектов). 
Сравнить производительность с последовательным методом. 
Сравнивть производительность Thread и multiprocessing решений. 
Попробовать подобрать оптимальное число потоков/процессов. 
"""
import requests, time

# Однопоточный режим с помощью request
img_url = 'https://cataas.com/cat'
amount = 100
directory = "Eugene_Piuta/DZ_06/cats/"

def dowload(i):   
    file_name = "image_" + str(i) + ".jpg"
    file = directory + file_name
    try:
        response = requests.get(img_url, stream=True)
        if response.status_code == 200:
            with open(file, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
        else:
            # Обработка ошибок при невозможности загрузки изображения
         print(f'Загрузить изображение не удалось: код ответа сервера {response.status_code}')
    except Exception as e:
        print(f"Error downloading image {file}: {e}")

def one_thred_downloads():
    # Start time
    t1 = time.time()
    for i, v in enumerate(range(amount)):      

            # Download images
            dowload(i)

    # End time
    t2 = time.time()
    print(f"Застрачено время на скачивание {amount} картинок в однопоточном режиме: {(t2 - t1)} сек.")

#Многопоточный режим с помощью ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
def download_TPE(max_workers):
    # Start time
    t1 = time.time()
    with ThreadPoolExecutor(max_workers) as executor:
        # Запуск задачи асинхронно
        for i in range(amount): 
            executor.submit(dowload, i)
    # End time
    t2 = time.time()
    print(f"Застрачено время на скачивание {amount} картинок с помощью ThreadPoolExecutor({max_workers} workers): {(t2 - t1)} сек.")
    
 #Многопоточный режим с помощью multiprocessing   
from multiprocessing import Pool
def download_MPS(max_proc):
    # Start time
    t1 = time.time()  
    with Pool(max_proc) as pool:                  
        for i in range(amount):    
            pool.apply_async(dowload(i))
    # End time
    t2 = time.time()
    print(f"Застрачено время на скачивание {amount} картинок с помощью multiprocessing({max_proc} processes): {(t2 - t1)} сек.")    
  
      
one_thred_downloads()
#Застрачено время на скачивание 100 картинок в однопоточном режиме: 57.37357306480408 сек.
download_TPE(5)
#Застрачено время на скачивание 100 картинок с помощью ThreadPoolExecutor(5 workers): 16.674795866012573 сек.
download_TPE(10)
#Застрачено время на скачивание 100 картинок с помощью ThreadPoolExecutor(10 workers): 8.538207530975342 сек.
download_TPE(20)  
#Застрачено время на скачивание 100 картинок с помощью ThreadPoolExecutor(20 workers): 9.093093872070312 сек.
download_MPS(2)
#Застрачено время на скачивание 100 картинок с помощью multiprocessing(2 processes): 72.66896677017212 сек. 
download_MPS(5)
#Застрачено время на скачивание 100 картинок с помощью multiprocessing(5 processes): 67.82863140106201 сек.
download_MPS(10)
#Застрачено время на скачивание 100 картинок с помощью multiprocessing(10 processes): 66.83982443809509 сек.
download_MPS(14)
#Застрачено время на скачивание 100 картинок с помощью multiprocessing(14 processes): 64.53756046295166 сек.

"""
Выводы:
Самым быстрым решением нашей задачи оказалось использование многопоточности( лучшие результаты на ±10 потоках).
На удивление однопоточный режим оказался чуть быстрее многопроцессорного.
Увеличение количества процессов не дает ощутимой экономии времени решения задачи.
    
    
"""