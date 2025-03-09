FROM python:3.12.8

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY model.pkl .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1080

CMD ["python", "app.py"]
