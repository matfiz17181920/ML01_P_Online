FROM python:3.9-slim

 
# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app


# Копируем файлы проекта в рабочую директорию
COPY . /app

# Устанавливаем зависимости
RUN pip3 --no-cache-dir install -r requirements.txt
RUN apt-get update && apt-get install -y curl

# Указываем порт, который будет слушать контейнер
EXPOSE 5000

# Запускаем приложение
CMD ["python", "app.py"]
