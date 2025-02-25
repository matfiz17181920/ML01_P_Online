# Используем официальный образ Python как базовый образ
FROM python:3.9-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файлы проекта в рабочую директорию
COPY . /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir flask scikit-learn joblib

# Указываем порт, который будет слушать контейнер
EXPOSE 5000

# Запускаем приложение
CMD ["python", "app_titanic2.py"]