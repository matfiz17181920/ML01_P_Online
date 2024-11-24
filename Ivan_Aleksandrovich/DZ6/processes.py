
""" В данном файле реализована загрузка файлов различным числом процессов, так как в Jupiter это сделать не удалось.
"""

import urllib.request
import os
import time
import multiprocessing as mp
import requests
from bs4 import BeautifulSoup

url = "https://unsplash.com/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
image_tags = soup.find_all('img')
image_links = [img['src'] for img in image_tags if 'src' in img.attrs]
absolute_links = [requests.compat.urljoin(url, link) for link in image_links] # Делаем ссылки абсолютными и неповторяющимися

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}. {e}')

def download (links, process_number):
    object_counter = 0
    dirname = "download"
    file_path = os.makedirs(dirname, exist_ok=True)
    for i, link in enumerate(links):
        try:
                response = requests.get(link)
                file_name = f"image_{i}_{process_number}.jpg"
                if not file_name.endswith('.jpg'): 
                    file_name += '.jpg'
                save_path = os.path.join(dirname, file_name)

                with open(save_path, 'wb') as file:
                    file.write(response.content)
                object_counter += 1
        except requests.RequestException as e:
           continue
    print(f"Процесс {process_number} загрузил {object_counter} объектов")



def run_with_process(num_process): # Запускаем процессы 
    processes = []
    last_item = 0
    batch_size = (len(absolute_links)//num_process) # Делим данные на части для разных процессов
    for i in range(num_process):
        process = mp.Process(target=download, args=(absolute_links[last_item:last_item+batch_size],i)) 
        processes.append(process)
        last_item += batch_size
        
    for process in processes:
        process.start() 
        
    for process in processes:    
        process.join() 


# Цикл, который будет запускать код с разным количеством процессов

if __name__ == "__main__":
    times_of_execution = {}
    for num_process in [2,4,8,10]: 
        print(f"\nЗапуск с {num_process} процессами...")
        start_time = time.time() 
        run_with_process(num_process)
        end_time = time.time() 

        times_of_execution[num_process] = end_time - start_time
            

    for key in times_of_execution.keys():
        print(f"Время выполнения с {key} процессами - {times_of_execution[key]} секунд")
            
